"""Which slots the NIXL dispatcher will re-point at a refilled rank's new owner."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.layers.moe.token_dispatcher.nixl import NixlEPBuffer
from sglang.test.test_utils import CustomTestCase


def _state(*, connected: int, local_rank: int):
    """The dispatcher state a refill runs against, with a recording buffer."""
    buffer = mock.MagicMock()
    buffer.rank = local_rank
    return SimpleNamespace(buffer=buffer, connected_ep_size=connected)


class TestNixlRefillRanks(CustomTestCase):
    """Replacing the endpoints of a slot whose process was replaced.

    Invariants, all of them the two asserts nixl's `disconnect_ranks` makes
    (`removed_rank != rank` and `_is_rank_connected(removed_rank)`), which abort
    the process rather than raise:

    - A connected slot is dropped before it is reconnected. `connect_ranks`
      skips a rank it already holds a connection to, so reconnecting alone
      leaves the departed process's endpoints in place.
    - Position in the connected range does not matter. `disconnect_ranks` erases
      by value and every per-rank structure behind it is keyed by global rank,
      so a hole in the middle is refillable exactly like one at the end.
    - A rank outside the connected range, or this process's own rank, is refused
      here instead of aborting inside nixl.
    """

    def test_tail_hole_is_dropped_then_reconnected(self):
        state = _state(connected=6, local_rank=0)

        with mock.patch.object(NixlEPBuffer, "_connect_ranks") as connect:
            NixlEPBuffer._refill_ranks(state, [4, 5])

        state.buffer.disconnect_ranks.assert_called_once_with([4, 5])
        connect.assert_called_once_with(state, [4, 5], tag="refill")

    def test_middle_hole_is_refilled_the_same_way(self):
        """The case the tail-only reading of the docs used to refuse."""
        state = _state(connected=64, local_rank=0)

        with mock.patch.object(NixlEPBuffer, "_connect_ranks") as connect:
            NixlEPBuffer._refill_ranks(state, list(range(8, 16)))

        state.buffer.disconnect_ranks.assert_called_once_with(list(range(8, 16)))
        connect.assert_called_once_with(state, list(range(8, 16)), tag="refill")

    def test_disconnect_precedes_connect(self):
        order = []
        state = _state(connected=8, local_rank=0)
        state.buffer.disconnect_ranks.side_effect = lambda r: order.append("disconnect")

        with mock.patch.object(
            NixlEPBuffer,
            "_connect_ranks",
            side_effect=lambda *a, **k: order.append("connect"),
        ):
            NixlEPBuffer._refill_ranks(state, [6, 7])

        self.assertEqual(order, ["disconnect", "connect"])

    def test_rank_beyond_the_connected_range_is_refused(self):
        state = _state(connected=8, local_rank=0)

        with mock.patch.object(NixlEPBuffer, "_connect_ranks") as connect:
            NixlEPBuffer._refill_ranks(state, [8, 9])

        state.buffer.disconnect_ranks.assert_not_called()
        connect.assert_not_called()

    def test_own_rank_is_refused(self):
        """nixl aborts on disconnecting yourself; never hand it that list."""
        state = _state(connected=8, local_rank=5)

        with mock.patch.object(NixlEPBuffer, "_connect_ranks") as connect:
            NixlEPBuffer._refill_ranks(state, [4, 5])

        state.buffer.disconnect_ranks.assert_not_called()
        connect.assert_not_called()

    def test_empty_rank_list_is_refused(self):
        state = _state(connected=8, local_rank=0)

        with mock.patch.object(NixlEPBuffer, "_connect_ranks") as connect:
            NixlEPBuffer._refill_ranks(state, [])

        state.buffer.disconnect_ranks.assert_not_called()
        connect.assert_not_called()

    def test_a_buffer_that_never_connected_refuses_everything(self):
        state = _state(connected=None, local_rank=0)

        with mock.patch.object(NixlEPBuffer, "_connect_ranks") as connect:
            NixlEPBuffer._refill_ranks(state, [0])

        state.buffer.disconnect_ranks.assert_not_called()
        connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
