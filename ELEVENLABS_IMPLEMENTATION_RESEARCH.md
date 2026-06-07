# ElevenLabs Implementation Research

Researched on June 6, 2026.

## Bottom line

The public evidence does **not** point to one universal "Eleven for voice, OpenAI for brain" stack.

Instead, the high-profile deployments split into two clear patterns:

1. **ElevenLabs as the voice layer inside someone else's platform**
   - Strong examples: Cisco Webex AI Agent, Convin, Regal, Twilio ConversationRelay.
   - In these cases, ElevenLabs is mainly the speech layer: TTS, low-latency audio, and sometimes STT.
   - The customer or partner keeps the orchestration, business logic, CRM/ERP integrations, and often the LLM choice.

2. **ElevenAgents as the hosted conversational platform**
   - Strong examples: Revolut, Klarna, Deliveroo, Customers Bank, Deutsche Telekom.
   - In these cases, ElevenLabs is doing more than voice: telephony integration, turn-taking, agent workflows, tools, evaluation, and sometimes the LLM too.
   - Even here, the customer may still keep their own orchestration and internal systems.

So the answer is:

- **Sometimes yes:** Eleven for voice, OpenAI for brain.
- **Sometimes no:** ElevenAgents is the whole runtime, with OpenAI selected natively inside ElevenAgents or replaced by another provider.
- **Often the real enterprise pattern is hybrid:** Eleven handles the speech runtime while the customer keeps account logic, policy logic, tool endpoints, and compliance controls.

## What is actually public

The table below separates **confirmed public facts** from **inference**.

| Company | Publicly confirmed | Most likely stack | Confidence |
|---|---|---|---|
| Revolut | On January 28, 2026, ElevenLabs said Revolut deployed **ElevenLabs Agents** as the first line of live voice support, with real-time language switching and secure connections to proprietary systems. ElevenLabs also says Revolut **kept control of orchestration and business logic** and kept its **in-house logic and models**. Source: [ElevenLabs case study](https://elevenlabs.io/blog/revolut/) | Telephony -> ElevenAgents STT/TTS/turn-taking -> Revolut orchestration layer -> Revolut proprietary systems and internal model stack. OpenAI is plausible, but **not publicly confirmed**. | High on the split between Eleven runtime and Revolut logic. Low on exact LLM vendor. |
| Klarna | On February 11, 2026, ElevenLabs said Klarna launched a voice AI agent **built with ElevenAgents** as the first line of US phone support for 35M customers. Source: [ElevenLabs case study](https://elevenlabs.io/blog/klarna) | Inbound phone support -> ElevenAgents platform -> Klarna support/account systems -> human escalation. LLM may be native inside ElevenAgents or external; **not public**. | High on ElevenAgents usage. Medium on the rest. |
| Deliveroo | On November 26, 2025, ElevenLabs said Deliveroo deployed an **ElevenLabs Agent** for outbound rider onboarding calls and restaurant phone verification. Source: [ElevenLabs case study](https://elevenlabs.io/blog/deliveroo/) | Outbound telephony -> ElevenAgents -> Deliveroo operations data / workflow tools -> CRM or internal ops systems. Exact LLM not public. | High on ElevenAgents + outbound calling. Medium on tool/backend details. |
| Customers Bank | On April 24, 2026, ElevenLabs said Customers Bank is deploying **ElevenAgents** across voice and chat, including a 24/7 phone/web/mobile support agent and a real-time coaching agent for live calls. Source: [ElevenLabs announcement](https://elevenlabs.io/blog/customers-bank-partnership) | Omnichannel support -> ElevenAgents -> bank systems via tools/APIs -> human escalation. Likely strong compliance wrapping. Exact LLM vendor not public. | High on ElevenAgents + banking workflows. Medium on underlying LLM/orchestration. |
| Deutsche Telekom | On March 2, 2026, ElevenLabs said Deutsche Telekom embedded AI voice agents **directly into network infrastructure**, activated during calls with "Hey Magenta". Source: [ElevenLabs announcement](https://elevenlabs.io/blog/deutsche-telekom-ai-call-assistant) | Telco network layer -> Eleven voice/agent runtime -> telecom-owned control plane and service integrations. This is not a normal app stack; it is carrier-grade network integration. | High on network integration. Low on component-level internals. |
| Meesho | On July 17, 2025, ElevenLabs said Meesho built a real-time voice agent using **ElevenLabs Text to Speech** and later described it as integrating **Conversational AI**. Source: [ElevenLabs case study](https://elevenlabs.io/blog/meesho) | Most likely: Meesho contact-center flow + Eleven voice runtime, with either their own control stack or early Eleven conversational runtime. Because the article emphasizes TTS quality, this looks more like **voice layer first** than full platform replacement. | Medium. Public writeup is less precise than Revolut/Klarna. |
| Cisco Webex AI Agent | On June 23, 2025, ElevenLabs said it powers the **voice technology** behind Cisco's Webex AI Agent, while Cisco handles LLM-based support, channel support, design tools, and integrations with CRM/ERP/HR. Source: [ElevenLabs case study](https://elevenlabs.io/blog/cisco-webex) | Cisco owns the agent platform and brain. ElevenLabs is the **voice layer**. This is the clearest example of "Eleven for voice, customer for brain/platform." | High. |
| Convin | On February 7, 2025, ElevenLabs said Convin added AI calling to its contact center platform using **ElevenLabs text to speech**. Source: [ElevenLabs case study](https://elevenlabs.io/blog/convin) | Convin platform + its own calling/orchestration stack + Eleven TTS. This is another clear voice-layer-only pattern. | High. |
| Regal | On September 5, 2025, ElevenLabs said Regal chose ElevenLabs for **Text to Speech** while Regal remained the AI-native contact center platform. Source: [ElevenLabs case study](https://elevenlabs.io/blog/regal) | Regal owns contact center, workflows, and customer integrations. Eleven powers voice output. | High. |
| Twilio ConversationRelay | On March 31, 2025, ElevenLabs said Twilio integrated ElevenLabs voices into **ConversationRelay**. Source: [ElevenLabs announcement](https://elevenlabs.io/blog/twilio-conversation-relay) | Twilio telephony + Twilio agent runtime + Eleven voice layer + whichever LLM/app logic the builder chooses. | High. |

## The strongest stack clues

### 1. Revolut is the best evidence for a hybrid enterprise pattern

Publicly confirmed by ElevenLabs:

- Revolut uses **ElevenLabs Agents** for live voice support.
- Revolut kept **orchestration and business logic**.
- ElevenLabs "slotted into" Revolut's **existing chat and digital AI stack**.
- Revolut kept its **in-house logic and models**.

That means the likely structure is:

1. Telephony or app voice entrypoint
2. ElevenAgents for STT, TTS, turn-taking, multilingual switching, session runtime
3. Revolut-owned orchestration and customer-account logic
4. Revolut-owned internal models and proprietary data systems

This is the closest public proof that large enterprises do **not** just hand the whole brain to ElevenLabs.

## 2. Cisco, Convin, and Regal are the clearest proof of "voice layer only"

All three public writeups point the same way:

- Cisco says Webex AI Agent uses LLMs, enterprise knowledge sources, CRM/ERP/HR integrations, while Eleven powers the voice.
- Convin added AI calling to an existing contact-center platform with ElevenLabs TTS.
- Regal built its AI Agent MVP on Regal's own platform and chose ElevenLabs for TTS realism.

This is a very important signal for your app, because your current architecture already matches this pattern:

- your backend owns policy logic and grounded business logic
- the voice layer is already separable
- ElevenLabs can slot in without forcing you to rewrite the brain

## 3. ElevenAgents is now a real orchestration platform, not just a voice API

ElevenLabs' current docs show that ElevenAgents supports:

- native OpenAI models such as GPT-5, GPT-4.1, and GPT-4o: [models doc](https://elevenlabs.io/docs/eleven-agents/customization/llm)
- **custom LLM endpoints** using OpenAI-compatible `/v1/chat/completions` or `/v1/responses`: [custom LLM doc](https://elevenlabs.io/docs/eleven-agents/customization/llm/custom-llm)
- telephony via **Twilio** and **SIP trunking**, including Exotel compatibility: [Twilio integration](https://elevenlabs.io/docs/eleven-agents/phone-numbers/twilio-integration/native-integration), [SIP trunking](https://elevenlabs.io/docs/eleven-agents/phone-numbers/sip-trunking)
- tool calling to external APIs: [server tools](https://elevenlabs.io/docs/eleven-agents/customization/tools/server-tools)
- workflows with transfer-to-number and human handoff: [workflows](https://elevenlabs.io/docs/eleven-agents/customization/agent-workflows), [transfer to number](https://elevenlabs.io/docs/eleven-agents/customization/tools/system-tools/transfer-to-number)
- built-in analysis/evaluation: [conversation analysis](https://elevenlabs.io/docs/eleven-agents/customization/agent-analysis), [success evaluation](https://elevenlabs.io/docs/eleven-agents/customization/agent-analysis/success-evaluation)
- simulation testing: [simulate conversations](https://elevenlabs.io/docs/eleven-agents/guides/simulate-conversation)

That means there are really **three** viable enterprise stacks now.

## The three practical stack patterns

### Pattern A: ElevenLabs voice layer + your own brain

Best public matches:

- Cisco Webex AI Agent
- Convin
- Regal
- likely parts of Meesho

Stack:

1. Telephony or browser client
2. ElevenLabs STT/TTS runtime
3. Your server-side orchestration
4. OpenAI or another LLM
5. Your tools, CRM, ERP, policy engine, analytics

When to use it:

- You already have a strong backend
- You need custom policy guardrails
- You do not want vendor lock-in on reasoning
- You want ElevenLabs mainly for voice quality and turn-taking

### Pattern B: ElevenAgents + your tools + native OpenAI model inside ElevenAgents

Best public matches:

- likely Klarna
- likely Deliveroo
- likely Customers Bank

Stack:

1. Telephony/web/mobile
2. ElevenAgents runtime
3. OpenAI selected in ElevenAgents model settings
4. Tool/webhook calls into your backend systems
5. Built-in evals, workflows, conversation logs

When to use it:

- You want faster implementation
- You are okay with Eleven owning more runtime behavior
- You still need your business systems via APIs

### Pattern C: ElevenAgents + your custom LLM endpoint

Best public match:

- Revolut is the strongest public clue

Stack:

1. Telephony/web/mobile
2. ElevenAgents runtime
3. Custom LLM endpoint on your server
4. Your own OpenAI calls, tool routing, guardrails, memory, policy layer
5. Your internal systems

When to use it:

- You want Eleven's telephony, STT/TTS, turn-taking, and evals
- But you want to keep the "brain" under your control
- You already have a grounded orchestration layer, as this repo does

## What this means for `DHL_POC`

Your repo is already much closer to the production patterns above than a typical demo app.

Current repo shape:

- Browser and phone-style frontend flow
- Backend-owned policy/orchestration in `backend/app.py`
- Data grounding in `backend/data/sap_mock.json`
- Existing separation between reasoning and voice
- Existing telephony path with Exotel hooks
- Existing alternative voice path in `frontend/src/lib/sarvam.ts`

So for this codebase, the best ElevenLabs migration paths are:

### Option 1: Closest to your current architecture

Keep your backend as the brain.

Swap voice I/O to **ElevenLabs Speech Engine**:

Browser or phone audio -> ElevenLabs speech runtime -> your backend -> OpenAI/tool layer -> ElevenLabs TTS back to caller

Why this matches your repo:

- same split you already use today
- easiest to preserve your guardrails
- easiest to preserve your SAP/data grounding
- closest to Cisco/Convin/Regal style deployments

### Option 2: Stronger phone-platform path

Use **ElevenAgents + custom LLM endpoint**:

Exotel/Twilio/SIP -> ElevenAgents -> your custom OpenAI-compatible endpoint -> your DHL logic/tools -> ElevenAgents voice back to caller

Why this is interesting:

- Eleven handles telephony runtime, turn-taking, and speech
- you keep business logic and tool execution
- closest public analogue is Revolut's "kept orchestration and business logic"

### Option 3: Fastest to launch, least custom

Use **ElevenAgents + native OpenAI model** and expose only tools/webhooks to your backend.

Why this is weaker for your case:

- simpler
- but less aligned with the heavy grounding and anti-hallucination strategy already present in `backend/app.py`

## My recommendation for this repo

If the goal is to learn from the enterprise patterns while minimizing risk:

1. Start with **Pattern A** for a fast technical spike.
2. If you want phone-grade production features, move to **Pattern C**.

That translates to:

1. Prove `ElevenLabs Speech Engine + OpenAI tool-using backend`.
2. If the spike is good, evaluate `ElevenAgents + custom LLM endpoint + Exotel SIP`.

That keeps your current strengths:

- grounded invoice logic
- deterministic tool layer
- human transfer rules
- DHL-specific guardrails

while replacing only the voice runtime.

## Full working example in this repo

I added a runnable example here:

- `examples/elevenlabs_openai_speech_engine/`

What it demonstrates:

- **ElevenLabs handles the voice session**
- **OpenAI handles the reasoning**
- **Local Python tools handle the business logic**
- the model must call a tool before it can quote invoice facts

This is the cleanest public implementation pattern that matches both:

- the enterprise evidence from Cisco / Convin / Regal
- your existing repo architecture

## Source list

- ElevenLabs voice agents page: https://elevenlabs.io/voice-agents
- Revolut case study: https://elevenlabs.io/blog/revolut/
- Klarna case study: https://elevenlabs.io/blog/klarna
- Deliveroo case study: https://elevenlabs.io/blog/deliveroo/
- Customers Bank announcement: https://elevenlabs.io/blog/customers-bank-partnership
- Deutsche Telekom announcement: https://elevenlabs.io/blog/deutsche-telekom-ai-call-assistant
- Meesho case study: https://elevenlabs.io/blog/meesho
- Cisco Webex case study: https://elevenlabs.io/blog/cisco-webex
- Convin case study: https://elevenlabs.io/blog/convin
- Regal case study: https://elevenlabs.io/blog/regal
- Twilio ConversationRelay announcement: https://elevenlabs.io/blog/twilio-conversation-relay
- ElevenAgents models: https://elevenlabs.io/docs/eleven-agents/customization/llm
- Custom LLM integration: https://elevenlabs.io/docs/eleven-agents/customization/llm/custom-llm
- Speech Engine overview: https://elevenlabs.io/docs/overview/capabilities/speech-engine
- Speech Engine quickstart: https://elevenlabs.io/docs/eleven-api/guides/cookbooks/speech-engine
- Twilio native integration: https://elevenlabs.io/docs/eleven-agents/phone-numbers/twilio-integration/native-integration
- SIP trunking: https://elevenlabs.io/docs/eleven-agents/phone-numbers/sip-trunking
- Server tools: https://elevenlabs.io/docs/eleven-agents/customization/tools/server-tools
- Workflows: https://elevenlabs.io/docs/eleven-agents/customization/agent-workflows
- Transfer to number: https://elevenlabs.io/docs/eleven-agents/customization/tools/system-tools/transfer-to-number
- Conversation analysis: https://elevenlabs.io/docs/eleven-agents/customization/agent-analysis
- Success evaluation: https://elevenlabs.io/docs/eleven-agents/customization/agent-analysis/success-evaluation
- Simulate conversations: https://elevenlabs.io/docs/eleven-agents/guides/simulate-conversation
