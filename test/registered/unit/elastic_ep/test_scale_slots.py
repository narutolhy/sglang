"""Unit tests for elastic-EP slot bookkeeping — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.elastic_ep.elastic_ep import (
    ElasticEPState,
    ElasticEPStateManager,
    ScaleCohortPlan,
    clear_scale_cohort,
    find_joining_cohort,
    plan_joining_cohort,
    register_scale_cohort,
)
from sglang.test.test_utils import CustomTestCase


def _make_state(*, effective_ep_size: int, max_ep_size: int) -> ElasticEPState:
    active = torch.ones(max_ep_size, dtype=torch.int32)
    active[effective_ep_size:] = 0
    state = ElasticEPState(
        active_ranks=active,
        last_active_ranks=active.clone(),
        active_ranks_cpu=active.clone(),
    )
    state.effective_ep_size = effective_ep_size
    state.original_ep_size = effective_ep_size
    return state


class TestElasticScaleSlots(CustomTestCase):
    """Slot arithmetic behind runtime scale-up and hole refill.

    Invariants:
    - A pending scale and a hole in the rank space are independent conditions.
    - A cohort may extend the rank space, refill holes, or straddle both.
    - A cohort may never claim a slot whose rank is still serving.
    - A failed scale leaves every slot it touched refillable.
    """

    def setUp(self):
        self.addCleanup(setattr, ElasticEPStateManager, "_instance", None)

    def _install(self, state: ElasticEPState) -> None:
        ElasticEPStateManager._instance = state

    def _kill(self, state: ElasticEPState, ranks) -> None:
        for rank in ranks:
            state.active_ranks[rank] = 0
        state.sync_active_to_cpu()

    # --- the two conditions is_scaling() used to conflate -----------------

    def test_holes_are_not_a_pending_scale(self):
        state = _make_state(effective_ep_size=40, max_ep_size=64)
        self._install(state)
        self._kill(state, range(32, 40))

        self.assertTrue(ElasticEPStateManager.has_inactive_ranks())
        self.assertFalse(ElasticEPStateManager.is_scale_pending())
        self.assertTrue(ElasticEPStateManager.is_scaling())
        self.assertEqual(
            ElasticEPStateManager.get_inactive_ranks(), list(range(32, 40))
        )

    def test_pending_scale_without_holes(self):
        state = _make_state(effective_ep_size=32, max_ep_size=64)
        self._install(state)

        self.assertTrue(ElasticEPStateManager.request_scale(40))
        self.assertTrue(ElasticEPStateManager.is_scale_pending())
        self.assertFalse(ElasticEPStateManager.has_inactive_ranks())

    # --- where a joining cohort may start ---------------------------------

    def test_slot_candidates_offer_holes_before_the_top(self):
        state = _make_state(effective_ep_size=64, max_ep_size=64)
        self._install(state)
        self._kill(state, [40, 41, 42, 43, 56, 57])

        self.assertEqual(
            ElasticEPStateManager.get_scale_slot_candidates(), [40, 56, 64]
        )

    def test_slot_candidates_are_run_starts_only(self):
        """A cohort owns consecutive ranks, so mid-run offsets are not offered."""
        state = _make_state(effective_ep_size=16, max_ep_size=16)
        self._install(state)
        self._kill(state, [4, 5, 6])

        self.assertEqual(ElasticEPStateManager.get_scale_slot_candidates(), [4, 16])

    # --- resolving a cohort against the rank space ------------------------

    def test_append_only_growth_keeps_previous_behaviour(self):
        state = _make_state(effective_ep_size=32, max_ep_size=64)
        self._install(state)

        slots, resulting = ElasticEPStateManager.resolve_scale_slots(32, 8)
        self.assertEqual(slots, list(range(32, 40)))
        self.assertEqual(resulting, 40)

    def test_refilling_holes_leaves_the_ep_size_unchanged(self):
        state = _make_state(effective_ep_size=64, max_ep_size=64)
        self._install(state)
        self._kill(state, range(40, 48))

        slots, resulting = ElasticEPStateManager.resolve_scale_slots(40, 8)
        self.assertEqual(slots, list(range(40, 48)))
        self.assertEqual(resulting, 64)

    def test_cohort_may_straddle_the_top_hole_and_new_space(self):
        state = _make_state(effective_ep_size=40, max_ep_size=64)
        self._install(state)
        self._kill(state, [38, 39])

        slots, resulting = ElasticEPStateManager.resolve_scale_slots(38, 4)
        self.assertEqual(slots, [38, 39, 40, 41])
        self.assertEqual(resulting, 42)

    def test_cohort_cannot_claim_a_serving_rank(self):
        state = _make_state(effective_ep_size=64, max_ep_size=64)
        self._install(state)
        self._kill(state, [40, 41])

        self.assertIsNone(ElasticEPStateManager.resolve_scale_slots(40, 4))

    # --- commit / rollback ------------------------------------------------

    def test_commit_activates_only_the_matched_slots(self):
        state = _make_state(effective_ep_size=64, max_ep_size=64)
        self._install(state)
        self._kill(state, list(range(40, 48)) + [56])

        ElasticEPStateManager.request_scale(64)
        plan = ScaleCohortPlan(
            rank_offset=40, slots=list(range(40, 48)), resulting_ep_size=64
        )
        self.assertTrue(ElasticEPStateManager.begin_scale(plan))
        self.assertEqual(ElasticEPStateManager.get_pending_slots(), list(range(40, 48)))

        ElasticEPStateManager.mark_scale_slots_active()
        ElasticEPStateManager.commit_scale()

        self.assertEqual(ElasticEPStateManager.get_effective_ep_size(), 64)
        # The hole this cohort did not cover must survive the commit.
        self.assertEqual(ElasticEPStateManager.get_inactive_ranks(), [56])

    def test_growth_commit_activates_the_new_ranks(self):
        state = _make_state(effective_ep_size=32, max_ep_size=64)
        self._install(state)

        ElasticEPStateManager.request_scale(40)
        plan = ScaleCohortPlan(
            rank_offset=32, slots=list(range(32, 40)), resulting_ep_size=40
        )
        ElasticEPStateManager.begin_scale(plan)
        ElasticEPStateManager.mark_scale_slots_active()
        ElasticEPStateManager.commit_scale()

        self.assertEqual(ElasticEPStateManager.get_effective_ep_size(), 40)
        self.assertEqual(ElasticEPStateManager.get_inactive_ranks(), [])
        self.assertFalse(ElasticEPStateManager.is_scaling())

    def test_failed_refill_leaves_the_slots_refillable(self):
        state = _make_state(effective_ep_size=64, max_ep_size=64)
        self._install(state)
        self._kill(state, range(40, 48))

        ElasticEPStateManager.request_scale(64)
        plan = ScaleCohortPlan(
            rank_offset=40, slots=list(range(40, 48)), resulting_ep_size=64
        )
        ElasticEPStateManager.begin_scale(plan)
        ElasticEPStateManager.mark_scale_slots_active()
        ElasticEPStateManager.fail_scale("joiner never showed up")

        self.assertEqual(
            ElasticEPStateManager.get_inactive_ranks(), list(range(40, 48))
        )
        self.assertFalse(ElasticEPStateManager.is_scale_pending())
        self.assertTrue(ElasticEPStateManager.request_scale(64))

    def test_failed_growth_rolls_the_new_ranks_back_out(self):
        state = _make_state(effective_ep_size=32, max_ep_size=64)
        self._install(state)

        ElasticEPStateManager.request_scale(40)
        plan = ScaleCohortPlan(
            rank_offset=32, slots=list(range(32, 40)), resulting_ep_size=40
        )
        ElasticEPStateManager.begin_scale(plan)
        ElasticEPStateManager.mark_scale_slots_active()
        ElasticEPStateManager.fail_scale("cohort died mid-join")

        self.assertEqual(ElasticEPStateManager.get_effective_ep_size(), 32)
        self.assertEqual(ElasticEPStateManager.get_inactive_ranks(), [])
        self.assertEqual(state.active_ranks[32:40].sum().item(), 0)


class TestJoinerStateInit(CustomTestCase):
    """The rank mask a joining process starts from.

    Invariants:
    - A scale joiner is not part of the world it joins, so it starts knowing
      only itself; claiming the primary's ranks would report them as its own.
    - A recover joiner is launched with the world it rejoins, so the healthy
      mask `init` built for it is already right and must survive. Zeroing it
      here only worked while `reset` re-activated everything below the
      effective size, which it no longer does -- the joiner came out believing
      every other rank had departed.
    """

    def setUp(self):
        self.addCleanup(setattr, ElasticEPStateManager, "_instance", None)

    def _init(self, *, mode: str, my_rank: int, world_size: int, max_ep_size: int):
        """Run the real _init_joiner_state over the state `init` would have built."""
        active = torch.ones(max_ep_size, dtype=torch.int32)
        active[world_size:] = 0
        state = ElasticEPState(
            active_ranks=active,
            last_active_ranks=active.clone(),
            active_ranks_cpu=active.clone(),
        )
        state.effective_ep_size = world_size
        state.original_ep_size = world_size
        ElasticEPStateManager._instance = state

        with (
            mock.patch.multiple(
                "sglang.srt.elastic_ep.elastic_ep",
                get_exec=mock.Mock(
                    return_value=SimpleNamespace(moe=SimpleNamespace(ep_join_mode=mode))
                ),
                get_parallel=mock.Mock(
                    return_value=SimpleNamespace(
                        ep_join_world_size=None,
                        ep_join_rank_offset=world_size,
                        tp_size=2,
                        elastic_ep_initial_size=None,
                    )
                ),
            ),
            mock.patch.object(torch.distributed, "get_rank", return_value=my_rank),
        ):
            ElasticEPStateManager._init_joiner_state(state)
        return state

    def test_recover_joiner_keeps_the_world_it_rejoins(self):
        state = self._init(mode="recover", my_rank=5, world_size=8, max_ep_size=16)

        self.assertEqual(ElasticEPStateManager.get_inactive_ranks(), [])
        self.assertFalse(ElasticEPStateManager.is_scaling())
        self.assertEqual(state.effective_ep_size, 8)

    def test_recover_joiner_survives_reset(self):
        """`reset` runs at the end of the rejoin path and must be a no-op there."""
        state = self._init(mode="recover", my_rank=5, world_size=8, max_ep_size=16)

        state.reset()

        self.assertEqual(ElasticEPStateManager.get_inactive_ranks(), [])
        self.assertEqual(state.active_ranks[8:].sum().item(), 0)

    def test_scale_joiner_starts_knowing_only_itself(self):
        state = self._init(mode="scale", my_rank=8, world_size=8, max_ep_size=16)

        # effective is now the world it is joining (offset 8 + tp 2), and only
        # its own rank is claimed inside it.
        self.assertEqual(state.effective_ep_size, 10)
        self.assertEqual(
            [r for r, a in enumerate(state.active_ranks.tolist()) if a], [8]
        )
        self.assertTrue(state.has_scaled)

    def test_reset_does_not_claim_a_departed_rank_is_healthy(self):
        """Why the recover path cannot rebuild its mask through `reset`."""
        state = _make_state(effective_ep_size=8, max_ep_size=16)
        ElasticEPStateManager._instance = state
        state.active_ranks[3] = 0
        state.sync_active_to_cpu()

        state.reset()

        self.assertEqual(ElasticEPStateManager.get_inactive_ranks(), [3])


class _FakeStore:
    def __init__(self):
        self.data = {}

    def set(self, key, value):
        self.data[key] = value

    def check(self, keys):
        return all(key in self.data for key in keys)

    def get(self, key):
        return self.data[key]

    def delete_key(self, key):
        self.data.pop(key, None)


class TestScaleCohortAnnouncements(CustomTestCase):
    """Announcement matching between a joining cohort and the primary.

    Invariants:
    - A cohort is matched by the slot offset it announced.
    - A consumed announcement is cleared, so the same offset can be refilled
      later without matching the departed cohort's stale key.
    """

    def setUp(self):
        self.addCleanup(setattr, ElasticEPStateManager, "_instance", None)
        patcher = mock.patch(
            "sglang.srt.elastic_ep.elastic_ep.get_global_tcp_store",
            return_value=_FakeStore(),
        )
        self.store = patcher.start().return_value
        self.addCleanup(patcher.stop)

    def test_growth_cohort_is_matched_at_the_top(self):
        ElasticEPStateManager._instance = _make_state(
            effective_ep_size=32, max_ep_size=64
        )
        register_scale_cohort(32, 8)

        self.assertEqual(find_joining_cohort(), (32, 8))
        plan = plan_joining_cohort(40)
        self.assertIsNone(plan.error)
        self.assertEqual(plan.slots, list(range(32, 40)))

    def test_target_size_mismatch_is_reported_not_raised(self):
        ElasticEPStateManager._instance = _make_state(
            effective_ep_size=32, max_ep_size=64
        )
        register_scale_cohort(32, 8)

        plan = plan_joining_cohort(48)
        self.assertIsNotNone(plan.error)
        self.assertEqual(plan.slots, [])

    def test_cleared_announcement_stops_matching(self):
        """A stale key would admit ranks whose replacement has not arrived."""
        state = _make_state(effective_ep_size=64, max_ep_size=64)
        ElasticEPStateManager._instance = state
        for rank in range(40, 48):
            state.active_ranks[rank] = 0
        state.sync_active_to_cpu()

        register_scale_cohort(40, 8)
        self.assertEqual(find_joining_cohort(), (40, 8))

        clear_scale_cohort(40)
        self.assertIsNone(find_joining_cohort())


if __name__ == "__main__":
    unittest.main()
