from __future__ import annotations

import os
import unittest

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
        self.assertEqual(payload["StatusCallbackEvents[0]"], "terminal")

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

    def test_localized_sarvam_voice_prefers_language_specific_speaker(self) -> None:
        self.assertEqual(policy_app.localized_sarvam_voice("ratan", "hi-IN"), "shubh")
        self.assertEqual(policy_app.localized_sarvam_voice("priya", "bn-IN"), "roopa")
        self.assertEqual(policy_app.localized_sarvam_voice("ratan", "en-IN"), "ratan")

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
        self.assertIn("Yogesh", text)
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
