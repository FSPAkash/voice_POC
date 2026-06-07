# ElevenLabs Implementation Review

Reviewed on June 7, 2026 against the current repo state and the previously gathered ElevenLabs customer research.

## Overall take

Your **architectural direction is good**.

The main app is following the strongest public enterprise pattern from the research:

- **ElevenLabs for voice**
- **your own backend for orchestration and business logic**
- **OpenAI for transcription and/or reasoning**
- **your own telephony and tool layer**

That is closest to the public Cisco / Regal / Convin pattern, and parts of the Revolut pattern where the customer keeps orchestration and business logic.

What you have today is **not** an ElevenAgents-style deployment. It is a **custom voice pipeline with ElevenLabs TTS inside it**. That is a valid approach, but it means you still own more runtime complexity than the companies leaning on ElevenAgents / Speech Engine / SIP integrations.

## Findings

### 1. High: the browser ElevenLabs audio path is still unreliable in live runs

Evidence:

- The frontend still throws `audio decode failed: Offset is outside the bounds of the DataView` in `frontend/src/lib/voice.ts:396-448`.
- A recent live call artifact on June 7, 2026 shows the error repeating many times in the transcript log: `backend/data/call_log.jsonl:2`.

Why it matters:

- This is a launch-blocking reliability issue for the browser voice path.
- The architecture is sound, but if the client cannot decode streamed Eleven audio consistently, the user experience collapses before the “brain” matters.

Implication:

- Before any serious Eleven rollout, the audio framing / chunk-boundary / playback contract needs to be treated as a production issue, not just a polish item.

### 2. High: `transfer_to_human` is not a real transfer

Evidence:

- The tool currently only returns contact metadata; it does not perform an actual telephony handoff: `backend/app.py:4808-4815`.

Why it matters:

- In the public enterprise patterns, human escalation is operational, not cosmetic.
- Right now the agent can say it will transfer, but the system does not actually execute a SIP / PSTN / agent-desktop handoff.

Implication:

- This is a major gap if you want parity with the “live agentic call” story on the ElevenLabs customer pages.

### 3. High: the phone path is still demo-grade, single-session telephony

Evidence:

- Telephony is explicitly modeled as Exotel-only in bootstrap config: `backend/app.py:4913-4918`.
- Only one active phone demo call is allowed at a time: `backend/app.py:4993-5001`.

Why it matters:

- High-profile deployments are built for concurrent call handling, queueing, failover, and real transfer semantics.
- Your current phone implementation is a very solid demo bridge, but not yet a production telephony runtime.

Implication:

- If you want to resemble the bigger ElevenLabs deployments, you still need either:
  - ElevenAgents telephony / SIP, or
  - a much harder custom telephony hardening pass on your Exotel bridge.

### 4. Medium: the main app is using ElevenLabs only as TTS, not as a full conversational runtime

Evidence:

- The code explicitly frames the voice layer as `ElevenLabs (TTS) + OpenAI realtime (STT)`: `backend/app.py:124-126`.
- Session creation only hands back custom backend websocket endpoints: `backend/app.py:4937-4975`.
- The custom browser voice loop is handled through `/api/tts/stream` and `/api/stt/stream`: `backend/app.py:5343-5525`, `backend/app.py:5527-5715`.

Why it matters:

- This is not wrong. It is actually aligned with some strong enterprise examples.
- But it means you are **missing the major Eleven platform features** that other big deployments appear to benefit from:
  - native speech runtime
  - built-in turn-taking
  - built-in conversation analysis
  - built-in simulations/evals
  - native SIP / phone-number integrations
  - native custom-LLM endpoint support in ElevenAgents

Implication:

- You are on the “voice vendor inside our own runtime” path, not the “Eleven-powered agent platform” path.

### 5. Medium: the voice identity layer is still generic, not production-grade persona management

Evidence:

- Default placeholder voice IDs are still wired as fallbacks: `backend/app.py:133-146`.
- Persona mapping resolves to one static Eleven voice per persona key: `backend/app.py:281-310`.
- Language handling only passes a language hint; there is no per-language voice strategy: `backend/app.py:346-348`.

Why it matters:

- The high-quality ElevenLabs deployments are very likely using carefully QA’d voice choices, and in many cases custom or cloned voices.
- Your current mapping is practical for development, but it is not yet a differentiated production voice layer.

Implication:

- If the goal is “sounds like a real DHL India collections caller,” this still needs:
  - voice selection QA per language
  - clone strategy or professionally selected source voices
  - acceptance tests for Hindi / Hinglish / Bengali / Marathi delivery

### 6. Medium: the voice-session security model is lightweight

Evidence:

- `/api/session` mints a plain `session_id` and returns raw websocket paths: `backend/app.py:4937-4975`.
- The TTS websocket trusts a `hello` message with `session_id`, `voice`, and `language_code`: `backend/app.py:5444-5450`.
- The STT websocket similarly trusts the client-provided `session_id` and hints: `backend/app.py:5693-5700`.

Why it matters:

- Enterprise voice runtimes usually issue short-lived signed tokens, scoped session credentials, or provider-generated ephemeral tokens.
- Your API keys are server-side, which is good, but the session contract itself is still lightweight.

Implication:

- Fine for an internal POC, but weak for internet-exposed production traffic.

### 7. Medium: test coverage is strong for policy logic, but thin for the actual ElevenLabs voice pipe

Evidence:

- The backend tests pass and cover policy/config behavior, including voice ID resolution: `backend/tests/test_policy_engine.py:607-612`.
- I ran:
  - `python -m pytest backend\tests\test_policy_engine.py -q`
  - `python -m pytest backend\tests\test_cost_accounting.py -q`
- Result: `68 passed`.

Why it matters:

- The biggest current risk is in the voice transport and streaming edge cases, not the policy engine.
- There is no comparable test coverage around:
  - `/api/tts/stream`
  - Eleven chunk framing
  - browser playback robustness
  - Exotel media interaction with Eleven TTS under interruption

Implication:

- The codebase is better tested than many POCs, but the riskiest Eleven-specific pieces are still the least verified.

## What looks good

- The separation of concerns is solid: voice vendor and business logic are decoupled.
- Your backend remains the source of truth for grounding and policy, which is exactly how careful enterprise teams avoid hallucination risk.
- The Exotel bridge plus custom orchestration shows good ownership of the full call loop.
- Cost accounting and environment-driven voice mapping are already more mature than a typical “hackathon” integration.

## What you are missing if you want parity with the best public ElevenLabs deployments

1. A stable, proven browser voice stream without decode failures.
2. Real human handoff, not just transfer metadata.
3. Multi-call telephony concurrency and operational hardening.
4. Either ElevenAgents or Speech Engine adoption if you want native Eleven runtime features.
5. Real voice QA / cloning strategy for Indian collections personas.
6. Better session security for public exposure.
7. Eleven-specific regression coverage and voice-path load testing.

## Recommendation

If I map this directly to the research, your current implementation is best described as:

- **architecturally correct**
- **operationally incomplete**
- **closer to Cisco/Regal-style integration than to Revolut/Klarna-style deployment**

If your goal is the best near-term path, I would keep the current backend-owned brain and choose one of these next steps:

1. **Harden the current custom Eleven TTS pipeline**
   - best if you want maximum control
   - first priority is audio reliability, true transfer, and concurrency

2. **Move the telephony/runtime layer toward ElevenAgents + custom LLM endpoint**
   - best if you want to keep your brain but offload more speech/telephony complexity
   - closest to the strongest enterprise hybrid pattern from the research

## Verification performed

- Searched the repo for ElevenLabs-related implementation paths.
- Reviewed the main browser and telephony voice paths.
- Reviewed the Speech Engine example separately.
- Ran:
  - `python -m pytest backend\tests\test_policy_engine.py -q`
  - `python -m pytest backend\tests\test_cost_accounting.py -q`
- Observed passing tests plus a live call log artifact containing repeated browser audio decode failures.
