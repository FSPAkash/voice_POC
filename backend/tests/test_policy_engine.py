from __future__ import annotations

import base64
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from backend import app as policy_app


class PolicyEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_policy_mode = policy_app.POLICY_ENGINE_MODE
        policy_app.POLICY_ENGINE_MODE = "deterministic"

    def tearDown(self) -> None:
        policy_app.POLICY_ENGINE_MODE = self._orig_policy_mode

    def test_default_voice_persona_is_yogesh(self) -> None:
        persona = policy_app.persona_for_voice(None)
        self.assertEqual(persona["name"], "Yogesh")
        self.assertEqual(persona["gender"], "male")

    def test_default_male_voice_is_shubh(self) -> None:
        self.assertEqual(policy_app.DEFAULT_REALTIME_VOICE, "shubh")

    def test_hinglish_stt_prefers_auto_detect(self) -> None:
        self.assertEqual(policy_app.sarvam_stt_language_code("hinglish"), "unknown")
        self.assertEqual(policy_app.sarvam_stt_language_code("english"), "en-IN")

    def test_plain_english_detector_does_not_misclassify_line_by_line_batao(self) -> None:
        self.assertFalse(policy_app.is_plain_english("line by line batao"))

    def test_plain_english_detector_accepts_clear_english_confirmation(self) -> None:
        self.assertTrue(policy_app.is_plain_english("Yes, this is Anthony."))

    def test_plain_english_detector_accepts_invoice_question_in_english(self) -> None:
        self.assertTrue(policy_app.is_plain_english("Tell me about this invoice by line by line."))

    def test_explicit_language_request_handles_english_switch_without_crashing(self) -> None:
        self.assertEqual(
            policy_app.explicit_language_request_language_id("I don't speak Hindi, please speak English."),
            "english",
        )

    def test_explicit_language_request_detects_marathi(self) -> None:
        self.assertEqual(
            policy_app.explicit_language_request_language_id("\u092e\u0930\u093e\u0920\u0940\u0924 \u092c\u094b\u0932\u093e."),
            "marathi",
        )

    def test_explicit_language_request_detects_tamil(self) -> None:
        self.assertEqual(
            policy_app.explicit_language_request_language_id("\u0ba4\u0bae\u0bbf\u0bb4\u0bbf\u0bb2\u0bcd \u0baa\u0bc7\u0b9a\u0bc1\u0b99\u0bcd\u0b95\u0bb3\u0bcd."),
            "tamil",
        )

    def test_supported_render_language_id_keeps_marathi_and_tamil(self) -> None:
        self.assertEqual(policy_app.supported_render_language_id("marathi"), "marathi")
        self.assertEqual(policy_app.supported_render_language_id("tamil"), "tamil")

    def test_language_id_for_script_detects_marathi_markers(self) -> None:
        self.assertEqual(
            policy_app.language_id_for_script(
                "\u092e\u0932\u093e line by line invoice \u0938\u093e\u0902\u0917\u093e.",
                "hinglish",
                "hinglish",
            ),
            "marathi",
        )

    def test_bootstrap_reports_live_voice_and_model_config(self) -> None:
        client = policy_app.app.test_client()

        response = client.get("/api/bootstrap")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIsInstance(payload, dict)
        config = payload["config"]
        self.assertEqual(config["chat_model"], policy_app.CHAT_MODEL)
        self.assertEqual(config["supervisor_model"], policy_app.SUPERVISOR_MODEL)
        self.assertEqual(config["language_coach_model"], policy_app.LANGUAGE_COACH_MODEL)
        self.assertEqual(config["stt_mode"], policy_app.SARVAM_STT_MODE)
        self.assertEqual(config["sarvam_voice_preset"]["speaker"], policy_app.DEFAULT_REALTIME_VOICE)
        self.assertEqual(config["sarvam_voice_preset"]["id"], f"{policy_app.DEFAULT_REALTIME_VOICE}-collections")
        self.assertEqual(config["pricing_reference"]["sarvam"]["currency"], "INR")
        self.assertEqual(config["pricing_reference"]["openai_currency"], "USD")
        self.assertEqual(config["telephony"]["provider"], "exotel")
        self.assertEqual(config["telephony"]["stream_sample_rate"], policy_app.EXOTEL_STREAM_SAMPLE_RATE)

    def test_build_exotel_stream_url_uses_render_public_url(self) -> None:
        original = os.environ.get("RENDER_EXTERNAL_URL")
        try:
            os.environ["RENDER_EXTERNAL_URL"] = "https://dhl-poc.onrender.com"
            url = policy_app.build_exotel_stream_url("cost_session_phone_123")
        finally:
            if original is None:
                os.environ.pop("RENDER_EXTERNAL_URL", None)
            else:
                os.environ["RENDER_EXTERNAL_URL"] = original

        self.assertEqual(
            url,
            f"wss://dhl-poc.onrender.com{policy_app.EXOTEL_STREAM_PATH}?session_id=cost_session_phone_123&sample-rate={policy_app.EXOTEL_STREAM_SAMPLE_RATE}",
        )

    def test_normalize_https_base_url_adds_scheme_when_missing(self) -> None:
        normalized = policy_app._normalize_https_base_url("api.exotel.com", "https://api.in.exotel.com")

        self.assertEqual(normalized, "https://api.exotel.com")

    def test_build_exotel_connect_payload_targets_bidirectional_stream(self) -> None:
        payload = policy_app.build_exotel_connect_payload(
            to_number="+91 9136152622",
            caller_id="022-461-82014",
            stream_url="wss://dhl-poc.onrender.com/api/exotel/media?session_id=abc",
            status_callback_url="https://dhl-poc.onrender.com/api/exotel/status",
        )

        self.assertEqual(payload["From"], "+919136152622")
        self.assertEqual(payload["CallerId"], "02246182014")
        self.assertEqual(payload["StreamType"], "bidirectional")
        self.assertEqual(payload["StatusCallbackEvents[]"], "terminal")

    def test_exotel_basic_auth_header_matches_expected_scheme(self) -> None:
        original_key = policy_app.EXOTEL_API_KEY
        original_token = policy_app.EXOTEL_API_TOKEN
        try:
            policy_app.EXOTEL_API_KEY = "key123"
            policy_app.EXOTEL_API_TOKEN = "token456"
            header = policy_app.exotel_basic_auth_header()
        finally:
            policy_app.EXOTEL_API_KEY = original_key
            policy_app.EXOTEL_API_TOKEN = original_token

        self.assertEqual(header, "Basic a2V5MTIzOnRva2VuNDU2")

    def test_parse_exotel_call_sid_reads_xml_response(self) -> None:
        raw = """<?xml version="1.0" encoding="UTF-8"?>
<TwilioResponse>
 <Call>
  <Sid>2642ec17f2cf0921cb2a2d4022171a62</Sid>
 </Call>
</TwilioResponse>"""

        call_sid = policy_app.parse_exotel_call_sid(None, raw)

        self.assertEqual(call_sid, "2642ec17f2cf0921cb2a2d4022171a62")

    def test_wav_to_pcm_accepts_raw_linear16_payload(self) -> None:
        pcm = b"\x00\x01\x02\x03"

        converted, sample_rate = policy_app._wav_to_pcm(pcm, fallback_sample_rate=16000)

        self.assertEqual(converted, pcm)
        self.assertEqual(sample_rate, 16000)

    def test_looks_like_agent_echo_flags_replayed_prompt(self) -> None:
        echoed = policy_app.looks_like_agent_echo(
            "This is Yogesh from DHL Express India",
            "Good afternoon, this is Yogesh from DHL Express India. Am I speaking with Mr Anthony Gressive?",
        )

        self.assertTrue(echoed)

    def test_looks_like_agent_echo_keeps_real_customer_answer(self) -> None:
        echoed = policy_app.looks_like_agent_echo(
            "Yes, this is Anthony speaking.",
            "Good afternoon, this is Yogesh from DHL Express India. Am I speaking with Mr Anthony Gressive?",
        )

        self.assertFalse(echoed)

    def test_should_apply_language_switch_hint_ignores_single_syllable_noise(self) -> None:
        self.assertFalse(policy_app.should_apply_language_switch_hint("ம்"))
        self.assertFalse(policy_app.should_apply_language_switch_hint("जी"))

    def test_should_apply_language_switch_hint_allows_explicit_language_request(self) -> None:
        self.assertTrue(policy_app.should_apply_language_switch_hint("தமிழில் பேசுங்கள்"))

    def test_build_llm_grounding_snapshot_is_compact_json(self) -> None:
        payload = json.loads(policy_app.build_llm_grounding_snapshot("DHL001"))

        self.assertEqual(payload["customer"]["account_number"], "DHL001")
        self.assertEqual(payload["totals"]["total_outstanding_inr"], 57920)
        self.assertEqual(len(payload["invoices"]), 3)

    def test_phone_turn_commit_delay_waits_longer_for_short_fragment(self) -> None:
        short_delay = policy_app.phone_turn_commit_delay_seconds("हाँ")
        long_delay = policy_app.phone_turn_commit_delay_seconds("पर पैसा नहीं है")

        self.assertGreater(short_delay, long_delay)
        self.assertEqual(long_delay, policy_app.PHONE_TURN_COMMIT_DELAY_SECONDS)

    def test_phone_language_switch_signal_drops_short_unrenderable_script_noise(self) -> None:
        signal = policy_app.phone_language_switch_signal(
            "\u0ab9\u0abe \u0ab0\u0ac7 \u0aa4\u0acb",
            "gu-IN",
            "hinglish",
            "hinglish",
        )

        self.assertEqual(signal["action"], "drop")
        self.assertEqual(signal["reason"], "unrenderable_script_candidate")

    def test_phone_language_switch_signal_allows_substantive_supported_script_switch(self) -> None:
        signal = policy_app.phone_language_switch_signal(
            "\u0b8e\u0ba9\u0b95\u0bcd\u0b95\u0bc1 payment approval \u0b87\u0ba9\u0bcd\u0ba9\u0bc1\u0bae\u0bcd \u0b95\u0bbf\u0b9f\u0bc8\u0b95\u0bcd\u0b95\u0bb5\u0bbf\u0bb2\u0bcd\u0bb2\u0bc8",
            "ta-IN",
            "hinglish",
            "hinglish",
        )

        self.assertEqual(signal["action"], "switch")
        self.assertEqual(signal["candidate_language_id"], "tamil")

    def test_phone_session_confirms_short_supported_language_switch_on_repeat(self) -> None:
        session = policy_app.PhoneCallSession(
            session_id="cost_session_phone_test",
            account_number="DHL001",
            target_number="+919136152622",
            caller_id="02246182014",
            language_id="hinglish",
            voice=policy_app.DEFAULT_REALTIME_VOICE,
        )

        session._handle_final_transcript("\u0ba8\u0bbe\u0ba9\u0bcd \u0ba4\u0bae\u0bbf\u0bb4\u0bcd \u0ba4\u0bbe\u0ba9\u0bcd", "ta-IN")

        self.assertFalse(any(entry["role"] == "customer" for entry in session.transcript))
        self.assertEqual(session.active_language_id, "hinglish")

        session._handle_final_transcript("\u0ba8\u0bbe\u0ba9\u0bcd \u0ba4\u0bae\u0bbf\u0bb4\u0bcd \u0ba4\u0bbe\u0ba9\u0bcd", "ta-IN")
        timer = session._turn_commit_timer
        if timer is not None:
            timer.cancel()

        self.assertTrue(any(entry["role"] == "customer" for entry in session.transcript))
        self.assertEqual(session.active_language_id, "tamil")

    def test_phone_session_short_partial_needs_real_speech_before_barge_in(self) -> None:
        session = policy_app.PhoneCallSession(
            session_id="cost_session_phone_test",
            account_number="DHL001",
            target_number="+919136152622",
            caller_id="02246182014",
            language_id="hinglish",
            voice=policy_app.DEFAULT_REALTIME_VOICE,
        )
        session._current_response_id = "utt_demo"
        session._current_mark_name = "mark_utt_demo"
        session._current_response_text = "Am I speaking with Mr Anthony Gressive?"
        session._speech_seconds_since_flush = 0.1

        session._handle_partial_transcript("yes", "en-IN")

        self.assertEqual(session._current_response_id, "utt_demo")

        session._speech_seconds_since_flush = 0.35
        session._handle_partial_transcript("yes", "en-IN")

        self.assertIsNone(session._current_response_id)

    def test_phone_session_greeting_ignores_small_echo_interruptions(self) -> None:
        session = policy_app.PhoneCallSession(
            session_id="cost_session_phone_test",
            account_number="DHL001",
            target_number="+919136152622",
            caller_id="02246182014",
            language_id="hinglish",
            voice=policy_app.DEFAULT_REALTIME_VOICE,
        )
        session._greeting_started = True
        session.turn_number = 0
        session._current_response_id = "utt_demo"
        session._current_mark_name = "mark_utt_demo"
        session._current_response_text = "Good evening, main Yogesh DHL Express India se bol raha hoon."
        session._last_agent_speak_start_at = time.time()

        session._handle_partial_transcript("hello", "en-IN")
        session._handle_final_transcript("hello", "en-IN")

        self.assertEqual(session._current_response_id, "utt_demo")
        self.assertFalse(any(entry["role"] == "customer" for entry in session.transcript))

    def test_phone_session_greeting_allows_real_identity_reply_without_waiting(self) -> None:
        session = policy_app.PhoneCallSession(
            session_id="cost_session_phone_test",
            account_number="DHL001",
            target_number="+919136152622",
            caller_id="02246182014",
            language_id="hinglish",
            voice=policy_app.DEFAULT_REALTIME_VOICE,
        )
        session._greeting_started = True
        session.turn_number = 0
        session._current_response_id = "utt_demo"
        session._current_mark_name = "mark_utt_demo"
        session._current_response_text = "Good evening, main Yogesh DHL Express India se bol raha hoon."
        session._last_agent_speak_start_at = time.time()

        session._handle_final_transcript("yes this is anthony", "en-IN")
        timer = session._turn_commit_timer
        if timer is not None:
            timer.cancel()

        self.assertIsNone(session._current_response_id)
        self.assertTrue(any(entry["role"] == "customer" and entry["text"] == "yes this is anthony" for entry in session.transcript))

    def test_opening_purpose_uses_pending_invoices_language(self) -> None:
        customer = policy_app.get_customer("DHL001")
        invoices = policy_app.get_invoices("DHL001")

        english = policy_app.opening_purpose_text(customer, invoices, "english", policy_app.DEFAULT_REALTIME_VOICE)
        hinglish = policy_app.opening_purpose_text(customer, invoices, "hinglish", policy_app.DEFAULT_REALTIME_VOICE)

        self.assertIn("pending DHL invoices", english)
        self.assertIn("pending DHL invoices", hinglish)
        self.assertNotIn("My name is", english)
        self.assertNotIn("मेरा नाम", hinglish)
        self.assertNotIn("credit account", english)
        self.assertNotIn("DHL123456", english)
        self.assertNotIn("DHL123456", hinglish)

    def test_repeat_request_helper_catches_hindi_confusion_phrase(self) -> None:
        self.assertTrue(policy_app.looks_like_repeat_request("कुछ समझा नहीं"))
        self.assertTrue(policy_app.looks_like_repeat_request("I do not understand it."))

    def test_fast_deterministic_turn_does_not_blanket_all_first_turns(self) -> None:
        affirmative = [
            {"role": "assistant", "text": "Good morning, am I speaking with Anthony?"},
            {"role": "customer", "text": "Yes, speaking."},
        ]
        generic = [
            {"role": "assistant", "text": "Good morning, am I speaking with Anthony?"},
            {"role": "customer", "text": "Maybe later."},
        ]

        self.assertTrue(policy_app.should_use_fast_deterministic_turn(affirmative))
        self.assertFalse(policy_app.should_use_fast_deterministic_turn(generic))

    def test_phone_ambience_profile_starts_from_stronger_section(self) -> None:
        raw = policy_app.load_phone_ambience_pcm(policy_app.EXOTEL_STREAM_SAMPLE_RATE)
        processed, start_offset = policy_app.load_phone_ambience_profile(policy_app.EXOTEL_STREAM_SAMPLE_RATE)

        self.assertTrue(raw)
        self.assertTrue(processed)
        self.assertGreater(start_offset, 0)
        raw_head_rms = policy_app.pcm16_rms(raw[:1600])
        processed_start_rms = policy_app.pcm16_rms(processed[start_offset : start_offset + 1600])
        self.assertGreater(processed_start_rms, raw_head_rms)

    def test_phone_session_emits_idle_ambience_after_playback_completion(self) -> None:
        session = policy_app.PhoneCallSession(
            session_id="cost_session_phone_test",
            account_number="DHL001",
            target_number="+919136152622",
            caller_id="02246182014",
            language_id="hinglish",
            voice=policy_app.DEFAULT_REALTIME_VOICE,
        )
        sent_payloads: list[dict[str, object]] = []
        session._send_json = sent_payloads.append  # type: ignore[assignment]
        session.stream_sid = "demo_stream"
        session._current_tts_serial = 2
        session._current_response_id = "utt_demo"
        session._current_mark_name = "mark_utt_demo"
        session._current_response_text = "Greeting"

        completed = session._complete_active_playback(2, "mark_utt_demo", "timer_fallback")
        time.sleep(0.25)
        session._stop = True

        self.assertTrue(completed)
        self.assertTrue(any(payload.get("event") == "media" for payload in sent_payloads))
        media_payload = next(payload for payload in sent_payloads if payload.get("event") == "media")
        raw = base64.b64decode(str(media_payload["media"]["payload"]))
        self.assertGreater(policy_app.pcm16_rms(raw), 2500)

    def test_llm_mode_uses_fast_deterministic_path_for_line_by_line_request(self) -> None:
        messages = [
            {
                "role": "assistant",
                "text": "Thank you for confirming. My name is Yogesh and I am calling from DHL Express India. I am calling about your pending DHL invoices.",
            },
            {"role": "customer", "text": "Tell me about it line by line"},
        ]

        original_mode = policy_app.POLICY_ENGINE_MODE
        original_client = policy_app.OPENAI_CLIENT

        class ExplodingClient:
            def __getattr__(self, _: str):
                raise AssertionError("LLM client should not be used for fast deterministic turns")

        try:
            policy_app.POLICY_ENGINE_MODE = "llm"
            policy_app.OPENAI_CLIENT = ExplodingClient()
            text, tool_calls, usage_events, error = policy_app.run_chat_agent_turn(
                messages=messages,
                voice="shubh",
                account_number="DHL001",
                language_advice={"suggested_language_id": "english", "detected_language_id": "english"},
            )
        finally:
            policy_app.POLICY_ENGINE_MODE = original_mode
            policy_app.OPENAI_CLIENT = original_client

        self.assertIsNone(error)
        self.assertEqual(usage_events, [])
        self.assertIn("one at a time", text.lower())
        self.assertTrue(any(call["name"] == "get_invoices" for call in tool_calls))

    def test_repeat_request_branch_stays_short_and_on_tree(self) -> None:
        messages = [
            {
                "role": "assistant",
                "text": "Thank you for confirming. My name is Yogesh and I am calling from DHL Express India. I am calling about your pending DHL invoices.",
            },
            {"role": "customer", "text": "I do not understand it."},
        ]

        text, tool_calls, usage_events, error = policy_app.run_chat_agent_turn(
            messages=messages,
            voice="shubh",
            account_number="DHL001",
            language_advice={"suggested_language_id": "english", "detected_language_id": "english"},
        )

        self.assertIsNone(error)
        self.assertEqual(usage_events, [])
        self.assertIn("let me keep it simple", text.lower())
        self.assertNotIn("DHL123456", text)
        self.assertEqual(tool_calls, [])

    def test_phone_session_recent_barge_in_keeps_short_final_confirmation(self) -> None:
        session = policy_app.PhoneCallSession(
            session_id="cost_session_phone_test",
            account_number="DHL001",
            target_number="+919136152622",
            caller_id="02246182014",
            language_id="hinglish",
            voice=policy_app.DEFAULT_REALTIME_VOICE,
        )
        session._current_response_id = "utt_demo"
        session._current_mark_name = "mark_utt_demo"
        session._current_response_text = "Am I speaking with Mr Anthony Gressive?"
        session._speech_seconds_since_flush = 0.35
        session._handle_partial_transcript("yes", "en-IN")

        session._handle_final_transcript("yes", "en-IN")
        timer = session._turn_commit_timer
        if timer is not None:
            timer.cancel()

        self.assertTrue(any(entry["role"] == "customer" and entry["text"] == "yes" for entry in session.transcript))

    def test_phone_session_playback_fallback_finishes_turn_without_mark(self) -> None:
        session = policy_app.PhoneCallSession(
            session_id="cost_session_phone_test",
            account_number="DHL001",
            target_number="+919136152622",
            caller_id="02246182014",
            language_id="hinglish",
            voice=policy_app.DEFAULT_REALTIME_VOICE,
        )
        session._current_tts_serial = 3
        session._current_response_id = "utt_demo"
        session._current_mark_name = "mark_utt_demo"
        session._current_response_text = "Greeting"

        completed = session._complete_active_playback(3, "mark_utt_demo", "timer_fallback")

        self.assertTrue(completed)
        self.assertIsNone(session._current_response_id)
        self.assertEqual(session.turn_number, 1)

    def test_create_session_reuses_requested_cost_session_id(self) -> None:
        client = policy_app.app.test_client()

        response = client.post(
            "/api/session",
            json={
                "session_id": "cost_session_test_123",
                "language_id": "hinglish",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["session_id"], "cost_session_test_123")

    def test_prepare_sarvam_tts_text_improves_invoice_speech(self) -> None:
        spoken = policy_app.prepare_sarvam_tts_text(
            "Invoice DHL123456; Invoice DHL654321",
            "en-IN",
        )

        self.assertEqual(spoken, "Invoice DHL 123456. Invoice DHL 654321")

    def test_prepare_sarvam_tts_text_preserves_number_commas(self) -> None:
        spoken = policy_app.prepare_sarvam_tts_text(
            "Outstanding amount is INR 57,920.",
            "en-IN",
        )

        self.assertEqual(spoken, "Outstanding amount is INR 57,920.")

    def test_load_call_history_serializes_backend_log_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            original_path = policy_app.CALL_LOG_FILE
            policy_app.CALL_LOG_FILE = Path(tmpdir) / "call_log.jsonl"
            try:
                policy_app.append_jsonl(
                    policy_app.CALL_LOG_FILE,
                    {
                        "id": "call_demo_1",
                        "mode": "voice",
                        "disposition": "Invoice resend requested",
                        "duration_sec": 42,
                        "cost_usd": 0.12,
                        "total_units": 1234,
                        "summary": {"headline": "Demo call"},
                        "timestamp": "2026-06-03T05:08:20.603445+00:00",
                        "costs": {
                            "combined": {"estimated_cost_usd": 0.12, "total_tokens": 1234},
                            "agent": {"estimated_cost_usd": 0.1, "total_tokens": 1000},
                        },
                    },
                )
                history = policy_app.load_call_history()
            finally:
                policy_app.CALL_LOG_FILE = original_path

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["id"], "call_demo_1")
        self.assertEqual(history[0]["mode"], "voice")
        self.assertEqual(history[0]["summary"]["headline"], "Demo call")
        self.assertEqual(history[0]["durationSec"], 42)

    def test_localized_sarvam_voice_prefers_language_specific_speaker(self) -> None:
        self.assertEqual(policy_app.localized_sarvam_voice("ratan", "hi-IN"), "shubh")
        self.assertEqual(policy_app.localized_sarvam_voice("priya", "bn-IN"), "roopa")
        self.assertEqual(policy_app.localized_sarvam_voice("ratan", "en-IN"), "aditya")

    def test_sarvam_tts_pace_uses_english_override(self) -> None:
        self.assertEqual(policy_app.sarvam_tts_pace("en-IN", "shubh"), policy_app.SARVAM_TTS_PACE_ENGLISH)
        self.assertEqual(policy_app.sarvam_tts_pace("hi-IN", "shubh"), policy_app.SARVAM_TTS_PACE)

    def test_scrub_forbidden_payment_methods_preserves_cash_flow_language(self) -> None:
        cleaned = policy_app.scrub_forbidden_payment_methods(
            "I understand cash flow is tight right now."
        )

        self.assertEqual(cleaned, "I understand cash flow is tight right now.")

    def test_scrub_forbidden_payment_methods_preserves_check_with_your_boss(self) -> None:
        cleaned = policy_app.scrub_forbidden_payment_methods(
            "Please check with your boss and let me know tomorrow."
        )

        self.assertEqual(cleaned, "Please check with your boss and let me know tomorrow.")

    def test_scrub_forbidden_payment_methods_rewrites_explicit_cash_payment_instruction(self) -> None:
        cleaned = policy_app.scrub_forbidden_payment_methods(
            "You can make payment by cash tomorrow."
        )

        self.assertEqual(cleaned, "You can make payment via DHL MyBill tomorrow.")

    def test_chat_turn_route_accepts_messages_without_explicit_transcript(self) -> None:
        client = policy_app.app.test_client()

        response = client.post(
            "/api/chat/turn",
            json={
                "messages": [
                    {"role": "assistant", "text": "Opening line"},
                    {"role": "customer", "text": "line by line batao"},
                ],
                "account_number": "DHL001",
                "voice": policy_app.DEFAULT_REALTIME_VOICE,
                "language_advice": {"suggested_language_id": "hindi", "detected_language_id": "hindi"},
            },
        )

        self.assertEqual(response.status_code, 200)

    def test_customer_turn_route_switches_to_english_on_explicit_request(self) -> None:
        client = policy_app.app.test_client()

        response = client.post(
            "/api/turn/customer",
            json={
                "transcript": "I don't speak Hindi, please speak English.",
                "current_language_id": "hindi",
                "preferred_language_id": "hindi",
                "messages": [
                    {"role": "assistant", "text": "Hindi opening"},
                    {"role": "customer", "text": "I don't speak Hindi, please speak English."},
                ],
                "recent_transcript": [
                    {"role": "assistant", "text": "Hindi opening"},
                    {"role": "customer", "text": "I don't speak Hindi, please speak English."},
                ],
                "account_number": "DHL001",
                "voice": policy_app.DEFAULT_REALTIME_VOICE,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["advice"]["suggested_language_id"], "english")

    def test_collapse_trailing_customer_messages_replaces_fragments(self) -> None:
        messages = [
            {"role": "assistant", "text": "Opening line"},
            {"role": "customer", "text": "main anthony"},
            {"role": "customer", "text": "baat kar raha hoon"},
        ]

        collapsed = policy_app.collapse_trailing_customer_messages(
            messages,
            "main anthony baat kar raha hoon",
        )

        self.assertEqual(
            collapsed,
            [
                {"role": "assistant", "text": "Opening line"},
                {"role": "customer", "text": "main anthony baat kar raha hoon"},
            ],
        )

    def test_invoice_summary_line_does_not_repeat_first_for_every_invoice(self) -> None:
        invoice = {
            "invoice_no": "DHL123456",
            "amount": 13600,
            "currency": "INR",
            "due_date": "2026-01-31",
            "overdue_days": 60,
        }

        line = policy_app.invoice_summary_line(invoice, "english")

        self.assertIn("Invoice DHL123456", line)
        self.assertNotIn("first overdue invoice", line.lower())

    def test_run_chat_agent_turn_uses_deterministic_engine_when_forced(self) -> None:
        messages = [
            {
                "role": "assistant",
                "text": "Good evening, mera naam Yogesh hai aur main DHL Express India se bol raha hoon. Kya main Mr Anthony Gressive se baat kar raha hoon?",
            },
            {"role": "customer", "text": "Yes, I am Antony."},
        ]

        text, tool_calls, usage_events, error = policy_app.run_chat_agent_turn(
            messages=messages,
            voice="shubh",
            account_number="DHL001",
            language_advice={"suggested_language_id": "english", "detected_language_id": "english"},
        )

        self.assertIsNone(error)
        self.assertEqual(usage_events, [])
        self.assertIn("pending DHL invoices", text)
        self.assertNotIn("My name is", text)
        self.assertTrue(any(call["name"] == "get_invoices" for call in tool_calls))

    def test_hindi_script_identity_confirmation_advances_to_purpose(self) -> None:
        messages = [
            {
                "role": "assistant",
                "text": "Good evening, main Yogesh DHL Express India se bol raha hoon. Kya main Mr Anthony Gressive se baat kar raha hoon?",
            },
            {"role": "customer", "text": "हाँ, तुम एंथनी से बात कर रहे हो?"},
        ]

        text, tool_calls, usage_events, error = policy_app.run_chat_agent_turn(
            messages=messages,
            voice="shubh",
            account_number="DHL001",
            language_advice={"suggested_language_id": "hindi", "detected_language_id": "hindi"},
        )

        self.assertIsNone(error)
        self.assertEqual(usage_events, [])
        self.assertIn("outstanding", text.lower())
        self.assertIn("invoice", text.lower())
        self.assertTrue(any(call["name"] == "get_invoices" for call in tool_calls))

    def test_hindi_script_payment_question_recovers_into_invoice_flow(self) -> None:
        messages = [
            {
                "role": "assistant",
                "text": "Good evening, main Yogesh DHL Express India se bol raha hoon. Kya main Mr Anthony Gressive se baat kar raha hoon?",
            },
            {"role": "customer", "text": "अरे किस बात का पेमेंट?"},
        ]

        text, tool_calls, usage_events, error = policy_app.run_chat_agent_turn(
            messages=messages,
            voice="shubh",
            account_number="DHL001",
            language_advice={"suggested_language_id": "hindi", "detected_language_id": "hindi"},
        )

        self.assertIsNone(error)
        self.assertEqual(usage_events, [])
        self.assertIn("outstanding", text.lower())
        self.assertIn("invoice", text.lower())
        self.assertTrue(any(call["name"] == "get_invoices" for call in tool_calls))

    def test_hindi_invoice_detail_question_recovers_after_bad_payment_prompt(self) -> None:
        messages = [
            {
                "role": "assistant",
                "text": "Good evening, main Yogesh DHL Express India se bol raha hoon. Kya main Mr Anthony Gressive se baat kar raha hoon?",
            },
            {"role": "customer", "text": "\u0939\u093e\u0901, \u0924\u0941\u092e \u090f\u0902\u0925\u0928\u0940 \u0938\u0947 \u092c\u093e\u0924 \u0915\u0930 \u0930\u0939\u0947 \u0939\u094b?"},
            {
                "role": "assistant",
                "text": "Thank you for sharing that. Kya aap payment ke liye ek specific date confirm kar sakte hain, ideally next 2 business days ke andar?",
            },
            {
                "role": "customer",
                "text": "\u0907\u0938 \u092c\u093e\u0924 \u0915\u093e \u092a\u0947\u092e\u0947\u0902\u091f \u0915\u094c\u0928 \u0938\u0947 \u0907\u0928\u0935\u0949\u0907\u0938\u0947\u0938 \u0935\u094b \u0924\u094b \u092c\u0924\u093e\u0913\u0964",
            },
        ]

        text, tool_calls, usage_events, error = policy_app.run_chat_agent_turn(
            messages=messages,
            voice="shubh",
            account_number="DHL001",
            language_advice={"suggested_language_id": "hindi", "detected_language_id": "hindi"},
        )

        self.assertIsNone(error)
        self.assertEqual(usage_events, [])
        self.assertIn("DHL123456", text)
        self.assertIn("DHL654321", text)
        self.assertIn("DHL332241", text)
        self.assertNotIn("Thank you for sharing that", text)
        self.assertTrue(any(call["name"] == "get_invoices" for call in tool_calls))

    def test_vague_payment_commitment_prompts_for_exact_date(self) -> None:
        messages = [
            {
                "role": "assistant",
                "text": "Good evening Mr Anthony Gressive. I am calling regarding your overdue DHL credit account. The total outstanding amount is INR 57,920 across 3 invoices.",
            },
            {"role": "customer", "text": "We will pay soon."},
        ]

        text, tool_calls, usage_events, error = policy_app.run_chat_agent_turn(
            messages=messages,
            voice="shubh",
            account_number="DHL001",
            language_advice={"suggested_language_id": "english", "detected_language_id": "english"},
        )

        self.assertIsNone(error)
        self.assertEqual(usage_events, [])
        self.assertIn("exact payment date", text.lower())
        self.assertIn("2 business days", text.lower())
        self.assertEqual(tool_calls, [])

    def test_unclear_turn_after_purpose_probes_for_reason_instead_of_looping_payment_date(self) -> None:
        messages = [
            {
                "role": "assistant",
                "text": "Good evening Mr Anthony Gressive. I am calling regarding your overdue DHL credit account. The total outstanding amount is INR 57,920 across 3 invoices.",
            },
            {"role": "customer", "text": "Okay."},
        ]

        text, tool_calls, usage_events, error = policy_app.run_chat_agent_turn(
            messages=messages,
            voice="shubh",
            account_number="DHL001",
            language_advice={"suggested_language_id": "english", "detected_language_id": "english"},
        )

        self.assertIsNone(error)
        self.assertEqual(usage_events, [])
        self.assertIn("reason for the delay", text.lower())
        self.assertNotIn("specific payment date", text.lower())
        self.assertEqual(tool_calls, [])


if __name__ == "__main__":
    unittest.main()
