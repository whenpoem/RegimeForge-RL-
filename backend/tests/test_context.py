from __future__ import annotations

import unittest

import numpy as np

from backend.regime_lens.context import append_context_state, build_temporal_context, initialise_context_history


class TemporalContextTests(unittest.TestCase):
    def test_context_history_initialises_pads_and_appends_without_mutating_original(self) -> None:
        first = np.array([1.0, -1.0], dtype=np.float32)
        second = np.array([2.0, -2.0], dtype=np.float32)
        third = np.array([3.0, -3.0], dtype=np.float32)

        history = initialise_context_history(first, context_len=3)
        self.assertEqual(len(history), 3)
        np.testing.assert_allclose(build_temporal_context(history, context_len=3), np.tile(first, 3))

        trimmed = build_temporal_context([first, second, third], context_len=2)
        np.testing.assert_allclose(trimmed, np.concatenate([second, third]).astype(np.float32))

        updated = append_context_state(history, second)
        self.assertEqual(len(history), 3)
        np.testing.assert_allclose(np.asarray(history[-1]), first)
        np.testing.assert_allclose(build_temporal_context(updated, context_len=3), np.concatenate([first, first, second]))

        appended_again = append_context_state(updated, third)
        np.testing.assert_allclose(build_temporal_context(appended_again, context_len=3), np.concatenate([first, second, third]))


if __name__ == "__main__":
    unittest.main()
