"""DPBudget + DataParallelController dispatch tests.

The e2e counterpart, over the real scheduler load-report path, is
test/registered/disaggregation/test_disaggregation_dp_attention.py.

Fragility: scheduler tests bypass `DataParallelController.__init__` via
`__new__` and inject only the attrs the schedulers read (`workers`, `status`,
`send_failed`, `_last_reported_status`, `dp_active`, `max_dp_size`,
`_active_workers`, `_active_count_cache`, `round_robin_counter`, `dp_budget`).
Update `_make_controller` if a scheduler starts reading another attr.
`maybe_external_dp_rank_routing` is exercised as the real method, no mock. The
refresh-throttle tests inject `load_snapshot_reader` and `_last_refresh_time`
on top of those.
"""

import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgspec.structs
import zmq

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.data_parallel_controller import (
    DataParallelController,
    DPBudget,
    LoadBalanceMethod,
)
from sglang.srt.managers.load_snapshot import LoadSnapshot

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


_BASE_LOAD = msgspec.structs.replace(
    LoadSnapshot(dp_rank=0),
    max_total_num_tokens=4096,
    max_running_requests=128,
)


def _load(**overrides) -> LoadSnapshot:
    return msgspec.structs.replace(_BASE_LOAD, **overrides)


def _make_controller(dp_size: int) -> DataParallelController:
    """Bypass __init__; inject only the attrs dispatch methods read."""
    ctl = DataParallelController.__new__(DataParallelController)
    ctl.workers = [MagicMock(name=f"worker_{i}") for i in range(dp_size)]
    ctl.status = [True] * dp_size
    ctl.max_dp_size = dp_size
    ctl.launch_dp_size = dp_size
    ctl.dp_active = [True] * dp_size
    ctl.send_failed = [False] * dp_size
    ctl._last_reported_status = [True] * dp_size
    ctl._active_workers = list(range(dp_size))
    ctl._active_count_cache = dp_size
    ctl.control_message_step = 1
    ctl.round_robin_counter = 0
    ctl.dp_budget = DPBudget(dp_size=dp_size)
    return ctl


def _kill_worker(ctl: DataParallelController, slot: int) -> None:
    """Make one worker behave like a scheduler that has gone away.

    ZMQ raises Again on a PUSH whose peer is missing only once SNDTIMEO is set;
    without the timeout the real socket blocks forever, which is the failure this
    guards against.
    """
    ctl.workers[slot].send_pyobj.side_effect = zmq.Again()
    ctl.workers[slot].send.side_effect = zmq.Again()


def _req(routed_dp_rank=None, bootstrap_room=None, input_ids=None):
    """Req stand-in; SimpleNamespace avoids pinning to the Req dataclass schema."""
    return SimpleNamespace(
        routed_dp_rank=routed_dp_rank,
        bootstrap_room=bootstrap_room,
        input_ids=input_ids or [],
    )


class TestDPBudgetUpdateBudget(CustomTestCase):
    def test_maps_running_plus_waiting_to_total_requests(self):
        budget = DPBudget(dp_size=2)
        budget.update_budget(
            [
                _load(dp_rank=0, timestamp=1.0, num_running_reqs=3, num_waiting_reqs=2),
                _load(dp_rank=1, timestamp=1.0, num_running_reqs=5, num_waiting_reqs=1),
            ]
        )
        self.assertEqual(budget.total_requests, [5, 6])

    def test_maps_num_total_tokens_not_num_used_tokens(self):
        budget = DPBudget(dp_size=2)
        budget.update_budget(
            [
                _load(
                    dp_rank=0, timestamp=1.0, num_used_tokens=100, num_total_tokens=150
                ),
                _load(
                    dp_rank=1, timestamp=1.0, num_used_tokens=80, num_total_tokens=80
                ),
            ]
        )
        self.assertEqual(budget.total_tokens, [150, 80])

    def test_partial_update_only_affects_reported_rank(self):
        budget = DPBudget(dp_size=3)
        budget.update_budget(
            [
                _load(
                    dp_rank=0, timestamp=1.0, num_running_reqs=10, num_total_tokens=100
                ),
                _load(
                    dp_rank=1, timestamp=1.0, num_running_reqs=20, num_total_tokens=200
                ),
                _load(
                    dp_rank=2, timestamp=1.0, num_running_reqs=30, num_total_tokens=300
                ),
            ]
        )
        budget.update_budget(
            [
                _load(
                    dp_rank=1,
                    timestamp=2.0,
                    num_running_reqs=1,
                    num_waiting_reqs=1,
                    num_total_tokens=50,
                )
            ]
        )
        self.assertEqual(budget.total_requests, [10, 2, 30])
        self.assertEqual(budget.total_tokens, [100, 50, 300])


class TestDPBudgetDispatch(CustomTestCase):
    """DPBudget.dispatch picks a rank from current state and updates counters."""

    def test_total_requests_dispatch_picks_min_and_increments(self):
        budget = DPBudget(dp_size=3)
        budget.total_requests = [4, 2, 7]
        rank = budget.dispatch(LoadBalanceMethod.TOTAL_REQUESTS)
        self.assertEqual(rank, 1)
        self.assertEqual(
            budget.total_requests[1],
            3,
            "dispatch should increment chosen worker's request count",
        )

    def test_total_tokens_dispatch_applies_estimated_tokens(self):
        budget = DPBudget(dp_size=3)
        budget.total_tokens = [100, 50, 200]
        budget.total_requests = [0, 0, 0]
        rank = budget.dispatch(LoadBalanceMethod.TOTAL_TOKENS, estimated_tokens=30)
        self.assertEqual(rank, 1, "should pick worker with min total_tokens")
        self.assertEqual(
            budget.total_tokens[1],
            80,
            "dispatch should add estimated_tokens to chosen worker",
        )
        self.assertEqual(
            budget.total_requests[1],
            1,
            "dispatch should also increment request count",
        )

    def test_total_tokens_tie_breaks_on_total_requests(self):
        budget = DPBudget(dp_size=3)
        budget.total_tokens = [50, 50, 50]
        budget.total_requests = [4, 2, 7]
        rank = budget.dispatch(LoadBalanceMethod.TOTAL_TOKENS, estimated_tokens=10)
        self.assertEqual(
            rank, 1, "tie on total_tokens should fall back to min total_requests"
        )

    def test_dispatch_returns_none_for_methods_not_handled(self):
        """Round-robin and follow_bootstrap_room dispatch elsewhere; DPBudget
        only handles the load-aware variants."""
        budget = DPBudget(dp_size=3)
        self.assertIsNone(budget.dispatch(LoadBalanceMethod.ROUND_ROBIN))
        self.assertIsNone(budget.dispatch(LoadBalanceMethod.FOLLOW_BOOTSTRAP_ROOM))


class TestRoundRobinScheduler(CustomTestCase):
    def test_cycles_through_active_workers_in_order(self):
        ctl = _make_controller(dp_size=4)
        for _ in range(8):
            ctl.round_robin_scheduler(_req())
        # 8 reqs across 4 active workers — 2 each, in round-robin order
        for i, worker in enumerate(ctl.workers):
            self.assertEqual(worker.send_pyobj.call_count, 2, f"worker {i} call count")

    def test_first_dispatch_picks_worker_zero(self):
        ctl = _make_controller(dp_size=4)
        ctl.round_robin_scheduler(_req())
        ctl.workers[0].send_pyobj.assert_called_once()
        for i in (1, 2, 3):
            ctl.workers[i].send_pyobj.assert_not_called()

    def test_skips_inactive_workers(self):
        ctl = _make_controller(dp_size=4)
        ctl.status[1] = False
        ctl.status[3] = False
        for _ in range(6):
            ctl.round_robin_scheduler(_req())
        # Only workers 0 and 2 are active — should split 6 reqs evenly
        self.assertEqual(ctl.workers[0].send_pyobj.call_count, 3)
        ctl.workers[1].send_pyobj.assert_not_called()
        self.assertEqual(ctl.workers[2].send_pyobj.call_count, 3)
        ctl.workers[3].send_pyobj.assert_not_called()

    def test_routed_dp_rank_bypasses_counter(self):
        """External dp-rank routing must not advance the counter."""
        ctl = _make_controller(dp_size=4)
        ctl.round_robin_scheduler(_req(routed_dp_rank=2))
        ctl.workers[2].send_pyobj.assert_called_once()
        self.assertEqual(
            ctl.round_robin_counter,
            0,
            "external routing must not advance the round-robin counter",
        )
        # Subsequent round-robin req still lands on worker 0
        ctl.round_robin_scheduler(_req())
        ctl.workers[0].send_pyobj.assert_called_once()


class TestFollowBootstrapRoomScheduler(CustomTestCase):
    def test_dispatches_by_bootstrap_room_modulo(self):
        ctl = _make_controller(dp_size=4)
        for room, expected_rank in [
            (0, 0),
            (1, 1),
            (4, 0),
            (5, 1),
            (100, 0),
            (101, 1),
        ]:
            ctl.follow_bootstrap_room_scheduler(_req(bootstrap_room=room))
            ctl.workers[expected_rank].send_pyobj.assert_called()

    def test_requires_bootstrap_room(self):
        ctl = _make_controller(dp_size=4)
        with self.assertRaises(AssertionError):
            ctl.follow_bootstrap_room_scheduler(_req(bootstrap_room=None))

    def test_routed_dp_rank_bypasses_bootstrap_room(self):
        ctl = _make_controller(dp_size=4)
        ctl.follow_bootstrap_room_scheduler(_req(routed_dp_rank=3, bootstrap_room=1))
        ctl.workers[3].send_pyobj.assert_called_once()
        ctl.workers[1].send_pyobj.assert_not_called()


class TestTotalRequestsScheduler(CustomTestCase):
    def test_dispatches_to_min_request_worker(self):
        ctl = _make_controller(dp_size=4)
        ctl.dp_budget.total_requests = [5, 3, 1, 4]
        ctl.total_requests_scheduler(_req())
        ctl.workers[2].send_pyobj.assert_called_once()
        for i in (0, 1, 3):
            ctl.workers[i].send_pyobj.assert_not_called()
        self.assertEqual(
            ctl.dp_budget.total_requests[2],
            2,
            "DPBudget must record the dispatch by incrementing the counter",
        )

    def test_routed_dp_rank_bypasses_budget(self):
        ctl = _make_controller(dp_size=4)
        ctl.dp_budget.total_requests = [5, 3, 1, 4]
        ctl.total_requests_scheduler(_req(routed_dp_rank=0))
        ctl.workers[0].send_pyobj.assert_called_once()
        # DPBudget must not be touched when bypassed
        self.assertEqual(
            ctl.dp_budget.total_requests,
            [5, 3, 1, 4],
            "external routing must not mutate DPBudget state",
        )


class TestStatusAwarenessInconsistency(CustomTestCase):
    """Document a divergence: ``round_robin_scheduler`` skips workers whose
    ``status`` is False, but ``total_requests_scheduler`` /
    ``total_tokens_scheduler`` route purely by DPBudget — they do NOT
    consult ``self.status``. If a future change unifies this behaviour,
    this test will fail and force a reviewer to confirm intent."""

    def test_total_requests_ignores_status(self):
        ctl = _make_controller(dp_size=4)
        # Worker 2 is the global minimum AND marked inactive.
        ctl.dp_budget.total_requests = [5, 3, 1, 4]
        ctl.status[2] = False
        ctl.total_requests_scheduler(_req())
        # Current behaviour: still dispatches to the inactive worker.
        ctl.workers[2].send_pyobj.assert_called_once()


class TestRefreshLoadBudgetThrottle(CustomTestCase):
    @staticmethod
    def _controller_with_reader(dp_size, snapshots):
        ctl = _make_controller(dp_size)
        ctl.load_snapshot_reader = MagicMock()
        ctl.load_snapshot_reader.read_all.return_value = snapshots
        return ctl

    def test_throttled_refresh_spreads_a_burst_across_ranks(self):
        idle = [_load(dp_rank=i, timestamp=1.0, num_total_tokens=0) for i in range(4)]
        ctl = self._controller_with_reader(dp_size=4, snapshots=idle)
        # A refresh stamp in the future keeps every call inside the window, so
        # the burst runs entirely on speculative counters.
        ctl._last_refresh_time = time.perf_counter() + 3600.0

        for _ in range(8):
            ctl.refresh_load_budget()
            ctl.total_tokens_scheduler(_req(input_ids=[0] * 100))

        ctl.load_snapshot_reader.read_all.assert_not_called()
        self.assertEqual(
            ctl.dp_budget.total_tokens,
            [200, 200, 200, 200],
            "speculative increments should spread the burst evenly",
        )
        for i, worker in enumerate(ctl.workers):
            self.assertEqual(
                worker.send_pyobj.call_count, 2, f"worker {i} should get 2 of 8 reqs"
            )

    def test_refresh_outside_window_overwrites_speculative_increments(self):
        reported = [
            _load(dp_rank=0, timestamp=2.0, num_total_tokens=10),
            _load(dp_rank=1, timestamp=2.0, num_total_tokens=20),
        ]
        ctl = self._controller_with_reader(dp_size=2, snapshots=reported)
        ctl._last_refresh_time = 0.0  # window has long passed
        ctl.dp_budget.total_tokens = [999, 999]

        ctl.refresh_load_budget()

        ctl.load_snapshot_reader.read_all.assert_called_once()
        self.assertEqual(
            ctl.dp_budget.total_tokens,
            [10, 20],
            "a fresh snapshot must replace the speculative state",
        )
        self.assertGreater(ctl._last_refresh_time, 0.0)

    def test_unchanged_snapshot_does_not_reset_the_burst(self):
        frozen = [_load(dp_rank=i, timestamp=1.0, num_total_tokens=0) for i in range(2)]
        ctl = self._controller_with_reader(dp_size=2, snapshots=frozen)
        ctl._last_refresh_time = 0.0
        ctl.refresh_load_budget()  # adopts timestamp 1.0

        for _ in range(4):
            ctl.total_tokens_scheduler(_req(input_ids=[0] * 50))
        after_burst = list(ctl.dp_budget.total_tokens)

        ctl._last_refresh_time = 0.0  # let the next refresh through the throttle
        ctl.refresh_load_budget()  # same timestamp -> update_budget skips it

        self.assertEqual(
            after_burst, [100, 100], "burst should have spread over both ranks"
        )
        self.assertEqual(
            ctl.dp_budget.total_tokens,
            after_burst,
            "a stale-timestamp snapshot must not wipe the speculative state",
        )


class TestDispatchToDepartedWorker(CustomTestCase):
    """Routing when a scheduler process is gone.

    Invariants:
    - A departed worker never stalls dispatch. Its PUSH socket has no peer, and
      ZMQ blocks such a send forever, which would freeze the single-threaded
      controller for every rank and stop it from processing the rank-status
      updates that route around the fault.
    - A departed slot is probed once, not once per request.
    - Requests pinned to a specific rank fail loudly rather than move.
    - A republished status vector does not readmit a departed slot; a genuine
      recovery does.
    """

    def test_request_lands_on_a_healthy_worker(self):
        ctl = _make_controller(8)
        _kill_worker(ctl, 3)

        for _ in range(8):
            ctl.round_robin_scheduler(_req())

        self.assertEqual(ctl.workers[3].send_pyobj.call_count, 1)  # refused
        accepted = sum(
            worker.send_pyobj.call_count
            for slot, worker in enumerate(ctl.workers)
            if slot != 3
        )
        self.assertEqual(accepted, 8)  # every request still got delivered
        self.assertTrue(ctl.send_failed[3])
        self.assertNotIn(3, ctl._active_workers)

    def test_departed_worker_is_probed_once(self):
        """Otherwise every request pays the send timeout again."""
        ctl = _make_controller(8)
        _kill_worker(ctl, 3)

        for _ in range(40):
            ctl.round_robin_scheduler(_req())

        self.assertEqual(ctl.workers[3].send_pyobj.call_count, 1)

    def test_a_whole_departed_cohort_is_survived(self):
        """Round robin is sequential, so a contiguous cohort is hit in a run."""
        ctl = _make_controller(64)
        for slot in range(32, 40):
            _kill_worker(ctl, slot)
        ctl.round_robin_counter = 32

        for _ in range(64):
            ctl.round_robin_scheduler(_req())

        self.assertEqual(
            [i for i, failed in enumerate(ctl.send_failed) if failed],
            list(range(32, 40)),
        )
        self.assertEqual(len(ctl._active_workers), 56)

    def test_pinned_rank_is_not_silently_rerouted(self):
        ctl = _make_controller(8)
        _kill_worker(ctl, 3)

        with self.assertRaises(ValueError):
            ctl.maybe_external_dp_rank_routing(_req(routed_dp_rank=3))
        self.assertTrue(ctl.send_failed[3])

    def test_republished_status_does_not_readmit_a_departed_slot(self):
        ctl = _make_controller(8)
        _kill_worker(ctl, 3)
        for _ in range(8):
            ctl.round_robin_scheduler(_req())

        # The ranks have not observed the fault yet and still report everyone up.
        ctl.update_active_ranks(SimpleNamespace(status=[True] * 8))

        self.assertTrue(ctl.send_failed[3])
        self.assertNotIn(3, ctl._active_workers)

    def test_a_recovered_rank_is_readmitted(self):
        ctl = _make_controller(8)
        _kill_worker(ctl, 3)
        for _ in range(8):
            ctl.round_robin_scheduler(_req())

        down = [True] * 8
        down[3] = False
        ctl.update_active_ranks(SimpleNamespace(status=down))
        ctl.workers[3].send_pyobj.side_effect = None
        ctl.workers[3].send.side_effect = None
        ctl.update_active_ranks(SimpleNamespace(status=[True] * 8))

        self.assertFalse(ctl.send_failed[3])
        self.assertIn(3, ctl._active_workers)

    def test_broadcast_skips_a_departed_worker(self):
        ctl = _make_controller(8)
        _kill_worker(ctl, 3)

        ctl.send_to_all_workers("control")

        for slot, worker in enumerate(ctl.workers):
            self.assertEqual(worker.send_pyobj.call_count, 1, f"slot {slot}")
        self.assertTrue(ctl.send_failed[3])


if __name__ == "__main__":
    unittest.main()
