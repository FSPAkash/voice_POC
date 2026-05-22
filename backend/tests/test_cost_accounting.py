from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from backend import app as pricing_app


class CostAccountingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self._original_ledger_file = pricing_app.LEDGER_FILE
        pricing_app.LEDGER_FILE = Path(self._tmpdir.name) / "cost_ledger.json"
        self.addCleanup(self._restore_ledger_file)

    def _restore_ledger_file(self) -> None:
        pricing_app.LEDGER_FILE = self._original_ledger_file

    def test_realtime_cost_from_usage_distributes_cached_rollup(self) -> None:
        usage = {
            "input_tokens": 1000,
            "output_tokens": 150,
            "input_tokens_details": {
                "audio_tokens": 600,
                "text_tokens": 400,
                "cached_tokens": 250,
            },
            "output_tokens_details": {
                "audio_tokens": 50,
                "text_tokens": 100,
            },
        }

        cost, token_map = pricing_app.realtime_cost_from_usage("gpt-realtime-mini", usage)

        self.assertEqual(
            token_map,
            {
                "text_input_tokens": 300,
                "text_cached_input_tokens": 100,
                "text_output_tokens": 100,
                "audio_input_tokens": 450,
                "audio_cached_input_tokens": 150,
                "audio_output_tokens": 50,
            },
        )
        expected = (
            300 * 0.6 / 1_000_000
            + 100 * 0.06 / 1_000_000
            + 450 * 10.0 / 1_000_000
            + 150 * 0.3 / 1_000_000
            + 100 * 2.4 / 1_000_000
            + 50 * 20.0 / 1_000_000
        )
        self.assertTrue(math.isclose(cost, expected, rel_tol=0, abs_tol=1e-12))

    def test_transcription_cost_from_usage_backfills_prompt_text_tokens(self) -> None:
        usage = {
            "input_tokens": 1500,
            "output_tokens": 300,
            "input_tokens_details": {
                "audio_tokens": 1200,
            },
        }

        cost, token_map = pricing_app.transcription_cost_from_usage("gpt-4o-mini-transcribe", usage)

        self.assertEqual(
            token_map,
            {
                "audio_input_tokens": 1200,
                "text_input_tokens": 300,
                "text_output_tokens": 300,
            },
        )
        expected = (
            1200 * 3.0 / 1_000_000
            + 300 * 1.25 / 1_000_000
            + 300 * 5.0 / 1_000_000
        )
        self.assertTrue(math.isclose(cost, expected, rel_tol=0, abs_tol=1e-12))

    def test_text_cost_from_usage_tracks_cached_and_uncached_input(self) -> None:
        usage = {
            "input_tokens": 1000,
            "output_tokens": 250,
            "input_tokens_details": {
                "cached_tokens": 400,
            },
        }

        cost, token_map = pricing_app.text_cost_from_usage("gpt-4.1", usage)

        self.assertEqual(
            token_map,
            {
                "text_input_tokens": 600,
                "text_cached_input_tokens": 400,
                "text_output_tokens": 250,
            },
        )
        expected = (
            600 * 2.0 / 1_000_000
            + 400 * 0.5 / 1_000_000
            + 250 * 8.0 / 1_000_000
        )
        self.assertTrue(math.isclose(cost, expected, rel_tol=0, abs_tol=1e-12))

    def test_load_ledger_backfills_missing_nested_fields(self) -> None:
        pricing_app.write_json(
            pricing_app.LEDGER_FILE,
            {
                "agent": {
                    "model": "gpt-realtime-mini",
                    "events": 0,
                    "response_usage": {
                        "text_input_tokens": 0,
                        "text_cached_input_tokens": 0,
                        "text_output_tokens": 0,
                        "audio_input_tokens": 0,
                        "audio_cached_input_tokens": 0,
                        "audio_output_tokens": 0,
                        "estimated_cost_usd": 0.0,
                    },
                    "transcription_usage": {
                        "model": "gpt-4o-mini-transcribe",
                        "audio_input_tokens": 0,
                        "text_output_tokens": 0,
                        "estimated_cost_usd": 0.0,
                    },
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                },
                "supervisor": pricing_app.base_supervisor_ledger(),
                "language_coach": pricing_app.base_language_coach_ledger(),
                "updated_at": pricing_app.utc_now_iso(),
            },
        )

        ledger = pricing_app.load_ledger()

        self.assertIn("chat_agent", ledger)
        self.assertIn("processed_usage_event_ids", ledger)
        self.assertIn("session_id", ledger)
        self.assertEqual(ledger["agent"]["transcription_usage"]["text_input_tokens"], 0)

    def test_agent_response_events_are_idempotent_and_session_scoped(self) -> None:
        ledger = pricing_app.default_ledger(
            realtime_model="gpt-realtime-mini",
            transcription_model="gpt-4o-mini-transcribe",
        )
        pricing_app.write_json(pricing_app.LEDGER_FILE, ledger)
        session_id = ledger["session_id"]
        usage = {
            "input_tokens_details": {
                "audio_tokens": 200,
                "text_tokens": 100,
            },
            "output_tokens_details": {
                "audio_tokens": 50,
                "text_tokens": 25,
            },
        }

        first = pricing_app.record_agent_response_usage(
            "gpt-realtime-mini",
            usage,
            event_id="evt-1",
            session_id=session_id,
        )
        duplicate = pricing_app.record_agent_response_usage(
            "gpt-realtime-mini",
            usage,
            event_id="evt-1",
            session_id=session_id,
        )
        stale = pricing_app.record_agent_response_usage(
            "gpt-realtime-mini",
            usage,
            event_id="evt-2",
            session_id="cost_session_stale",
        )

        self.assertEqual(first["agent"]["events"], 1)
        self.assertEqual(first["agent"]["response_usage"], duplicate["agent"]["response_usage"])
        self.assertEqual(first["agent"]["estimated_cost_usd"], duplicate["agent"]["estimated_cost_usd"])
        self.assertEqual(duplicate["agent"]["estimated_cost_usd"], stale["agent"]["estimated_cost_usd"])
        self.assertEqual(duplicate["agent"]["events"], stale["agent"]["events"])

        reloaded = pricing_app.load_ledger()
        self.assertEqual(reloaded["processed_usage_event_ids"], ["evt-1"])


if __name__ == "__main__":
    unittest.main()
