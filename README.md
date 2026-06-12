# Evaluation & Test Strategy Charter

> **TEMPLATE NOTE (delete before submission):** This markdown mirrors the Celanese IT Charter Template v1.1 skeleton. Paste each section into the official `.docx` template downloaded from
> *IT Documentation Library → IT Documentation → 00 - Templates → All Documents*.
> Replace bracketed placeholders before submission.

---

**Document Number:** [Assign per IT Documentation Library convention]
**Version:** 0.1 (Draft)
**Release Date:** 2026-06-04
**Effective Date:** [YYYY-MM-DD after approval]
**Template Version:** 1.1

---

## Disclaimer

All product and company names which may be used within this document are trademarks ™ or registered ® trademarks of their respective holders. Use of them does not imply any affiliation with or endorsement by them.

The latest approved version of this document may always be found here:
http://teams.celanese.com/sites/ITDocs/IT%20Documentation/Forms/AllItems.aspx

Printed copies of this document are not authoritative. Only the latest digital copy located above is considered to be authoritative as of the effective date. This document supersedes any and all previous documents by the name or within the same version history as of its effective date.

---

## Document Attributes

### Keywords

- **Evaluation & Test Strategy** — the standards, methodologies, metrics, datasets, tooling, and governance gates that determine whether an AI solution at Celanese is fit to deploy and fit to remain deployed.
- **Fit to Deploy** — the criteria an AI solution must meet before promotion from pilot to production.
- **Fit to Remain Deployed** — the criteria an in-production AI solution must continue to meet (drift thresholds, monitoring SLAs, periodic re-evaluation) to stay in service.
- **Offline Evaluation** — pre-deployment evaluation against golden / holdout / regression datasets.
- **Online Evaluation** — production monitoring of live traffic against drift, quality, cost, and safety thresholds.
- **Adversarial Evaluation** — security & safety testing including prompt injection, jailbreaks, data exfiltration, bias amplification, output-handling exploits.
- **Eval-Hook Contract** — the telemetry / observability emit format every AI solution conforms to. Defined in §9 (Technology Architecture); consumed by Evaluation & Test Strategy.
- **Eval Gate** — a governance checkpoint in the AI lifecycle where evaluation results determine whether a solution advances to the next stage.
- **Profile** — same three profiles as §9 — Industrial / Operational, Customer / Commercial, Corporate / Productivity — each with its own evaluation emphasis.

### Comments

This is a v0.1 draft prepared for the Phase 2 review meeting on June 4, 2026. The strategy is presented **as the proposed standard** every Celanese AI solution must conform to — both at deployment and in production. Items requiring leadership decision before the deep-dive document are tagged **[OPEN]** inline.

### Charter at a Glance — What · Why · Where · When · Who · How

*Per Tele's directive (May 20 mail): "Please ensure to cover the What, Why, Where, When, Who and How (high-level) in the charter document."*

| Question | High-level answer |
|---|---|
| **What** | The **evaluation & test discipline** for Celanese AI — taxonomy, metrics, datasets, tooling, adversarial testing, production monitoring, and the governance gates that determine when a solution is *fit to deploy* and *fit to remain deployed*. |
| **Why** | To make Tele's mandate *"safely scaled"* and *"defined SLAs"* enforceable in practice — translate Responsible-AI principles, risk policies, and business SLAs into **measurable tests** across the entire AI lifecycle. Without evaluation discipline, deployment is opinion; with it, deployment is evidence. |
| **Where** | **Every AI solution at Celanese**, across all three profiles (Industrial / Operational, Customer / Commercial, Corporate / Productivity) — at every lifecycle stage (pilot → pre-deploy → production → continuous monitoring → revalidation). |
| **When** | Charter approval **June 4, 2026**.  Evaluation taxonomy v1 **Jul 2026**.  Metrics catalog v1 **Jul–Aug 2026**.  Eval harness slot decision **Aug 2026**.  Penetration Testing capability stand-up **Sep 2026**.  Full Evaluation & Test Strategy document **end-Sep 2026**.  Eval-gates live in §10 lifecycle **Q4 2026**. |
| **Who** | **Owners:** Sarath Mandadi & Aman Agarwal.  **Liaisons:** Technical Architecture (Mohan), AI Lifecycle (Nagesh, Sreeni), Responsible AI (Sam, Mohan), Data Architecture (Sreeni, Aman), Information Security, Application / Business-Unit reps per profile.  **Approvers:** Tele Fernandes; at least one IT Director per template requirement. |
| **How** | Six functions: **Evaluation Framework & Taxonomy · Metrics, Thresholds & SLAs · Evaluation Datasets & Tooling · Adversarial & Safety Testing · Production Monitoring, Drift & Continuous Improvement · Eval Gates & Lifecycle Integration.** **Calibrated for agentic workflows** — measures tool selection, reasoning, step sequencing, task completion, and policy compliance of every agent action · seven evaluation dimensions (Retrieval Quality · Generation Quality · Groundedness · Agentic Behavior · Business Validation · Operational · Safety & Policy) · **LLM-adaptation-aware** (Prompting vs RAG vs Fine-Tuning) · vendor-pluggable harness slot · profile-specific thresholds · eval-hook contract co-defined with §9 · governance gates co-owned with §10. |

---

## Version History

| Version | Date       | Author(s)                                  | Description                                          |
|---------|------------|--------------------------------------------|------------------------------------------------------|
| 0.1     | 2026-06-04 | Sarath Mandadi, Aman Agarwal               | Initial draft for Phase 2 review.                    |

---

## Ownership

The following roles/individuals are defined for this document:

- **Document Owner:** Tele S. Fernandes / Director, Advanced Analytics & Insights
- **Document Author:** Sarath Mandadi / [Role TBD]; Aman Agarwal / [Role TBD]

---

## Approvals

| Role                                         | Name                  | Date       |
|----------------------------------------------|-----------------------|------------|
| Director, Advanced Analytics & Insights      | Tele S. Fernandes     | [Pending]  |
| [IT Director — required for publication]     | Aaron Pryor (or designee) | [Pending] |
| [Director, Information Security]             | [Name TBD]            | [Pending]  |
| [Director, Quality / Risk]                   | [Name TBD]            | [Pending]  |
| [Director, Applications]                     | [Name TBD]            | [Pending]  |

> *Template requirement: at least one IT Director must approve before publication.*

---

## Evaluation & Test Strategy Definition

The **Evaluation & Test Strategy** group will **provide** the standards, methodologies, metrics, datasets, tooling, and governance gates that determine whether an AI solution at Celanese is **fit to deploy** and **fit to remain deployed** — across all three solution profiles and all lifecycle stages.

This group **defines what good looks like and how we measure it**. It does not own the lifecycle pipelines that execute the tests (that is §10 — AI Development Lifecycle), nor the Responsible-AI principles being tested for (those are §4 and §5), nor the telemetry contract that emits the data being evaluated (that is §9 — Technology Architecture). It owns the **strategy, the metrics, the methodology, the datasets, and the gates**.

**Calibrated for agentic workflows.** Celanese is building **agentic AI** — solutions where models reason, plan, retrieve grounding context, call tools, and take multi-step action against real systems (SAP, CRM, plant historians, MES, customer-facing surfaces). Classical-ML evaluation (single-prediction accuracy) and single-turn LLM evaluation (one prompt → one response) are necessary but insufficient. Evaluation & Test Strategy at Celanese must measure **tool-selection correctness, reasoning quality, step-sequence correctness, task-completion rate, hallucination & groundedness, and the policy compliance of every action the agent takes** — alongside the conventional dimensions of generation quality, retrieval quality, latency, and cost. Every metric in this charter has been chosen with agentic workflows in mind.

It is the responsibility of the Evaluation & Test Strategy group to:

1. Define and maintain the **Evaluation Taxonomy** — the canonical categories of evaluation: offline, online, adversarial, drift, performance, safety, bias, robustness, cost — and which categories apply to which profile and pattern.
2. Maintain the **Metrics, Thresholds & SLAs catalog** — for each category, the standard metrics, the acceptable thresholds, and the SLA bindings per profile. Translates Responsible-AI principles (§4, §5), risk policies (§6), and business SLAs into **measurable, testable criteria**.
3. Define **Evaluation Datasets & Tooling standards** — golden sets, regression sets, holdout sets, synthetic data, labeling standards, dataset governance, and the approved eval harness(es). Eval harness is treated as a **vendor-pluggable slot** in the §9 tool catalog.
4. Operate the **Adversarial & Safety Testing** function — penetration testing, prompt-injection testing, jailbreak resistance, data-exfiltration probes, bias amplification audits, output-handling exploits. Coordinates with §7 (Threat Landscape) and §11 (Security).
5. Define the **Production Monitoring, Drift & Continuous Improvement** methodology — online eval, drift detection, alert routing, periodic re-evaluation cadence, and the feedback loop from production telemetry into model improvement.
6. Define **Eval Gates** that hook into the §10 AI Development Lifecycle — when evaluation runs, what it blocks, what the exit criteria are, who approves an exception.
7. Co-define (with §9) the **eval-hook contract** — the telemetry every AI solution emits — so that what gets measured in production is consistent with what was measured pre-deployment.
8. Stay aligned with adjacent groups (Responsible AI, AI Lifecycle, Data Architecture, Technical Architecture, Information Security) so that evaluation criteria evolve with the rest of the framework.

> **Scope boundary statement:** Evaluation & Test Strategy defines the *standards and methodology* for evaluating AI solutions. It does not own the execution pipelines (§10), the principles being tested for (§4/§5), the data that flows into eval (§8), or the telemetry contract that emits the data (§9). It consumes from those peers and produces the measurement discipline they all need.

---

## Evaluation & Test Strategy Purpose

The Evaluation & Test Strategy group's purpose is to make Tele Fernandes' *AI specific goals* memo (Apr 22, 2026) **operational and enforceable** — specifically the directives to *"move AI solutions beyond experimentation by standardizing how models and agents are deployed, observed, and maintained in production"* and to introduce *"performance monitoring, drift tracking, cost visibility, and defined SLAs so AI solutions can be safely scaled."*

In plain terms: **without evaluation discipline, deployment is opinion. With it, deployment is evidence.**

Initial efforts will focus on:

- **Publishing the v1 Evaluation Taxonomy** — the canonical map: four tracks (offline · online · adversarial · periodic revalidation) × three profiles × five §9.2 patterns × the agentic-workflow dimensions (tool use, reasoning, step sequencing, task completion).
- **Standing up the v1 Metrics, Thresholds & SLAs catalog across seven evaluation dimensions** — Retrieval Quality (Recall@K, Precision@K, MRR), Generation Quality (Exact match, F1, semantic similarity, LLM-as-judge), Groundedness & Hallucination, Agentic Behavior (tool selection, reasoning, step sequence, task completion), Business Validation (effort reduction, satisfaction, acceptance rate), Operational (latency, token usage, API-call count, cost), Safety & Policy (PII, toxicity, brand, forbidden-action prevention).
- **Defining the LLM Adaptation Strategy ↔ Eval Mapping** — how the evaluation menu shifts between prompt-engineered, RAG-augmented, and fine-tuned solutions, so eval emphasis matches how the solution was built.
- **Confirming the eval-harness slot decision** — vendor-pluggable; candidates include Azure AI Studio Evaluations, PromptFlow, DeepEval, Ragas (RAG-specific), or a hybrid combining managed + open-source. Agentic-trace evaluation tooling considered separately (LangSmith, AgentOps, or custom).
- **Co-defining the eval-hook contract with §9** — what every AI solution must emit (including per-step agent traces) so that production monitoring is interpretable end-to-end.
- **Standing up the penetration testing capability** — including **agent-specific adversarial vectors**: tool misuse, forbidden-action attempts, multi-step plan injection, reasoning hijack, tool-argument injection.
- **Defining drift-detection methodology** — input drift, output drift, embedding drift, performance drift, **tool-use drift** (changes in agent tool-selection patterns over time).
- **Defining the eval gates** that promote a solution through the §10 lifecycle (pilot → pre-deploy → production → revalidation), with explicit pass criteria per gate per profile.

---

## Evaluation & Test Strategy Composition

The Evaluation & Test Strategy group will be comprised of, at a minimum, the following elements:

| Role                                              | Area of Responsibility                                                                                                       |
|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| Evaluation & Test Strategy Lead (Sarath Mandadi)  | Overall direction; taxonomy authorship; eval-hook contract alignment with §9; charter execution.                             |
| Evaluation & Test Strategy Lead (Aman Agarwal)    | Co-lead; metrics catalog, datasets, tooling, harness selection.                                                              |
| Technical Architecture Liaison (Mohan K. Samavedam) | §9 — eval-hook contract; L3 telemetry component; vector / knowledge store integration for eval datasets.                  |
| AI Lifecycle Liaison (Nagesh Cheripally / Sreeni Iyer) | §10 — eval-gate integration into pipelines; CI/CD hook design; deployment & release-gate execution.                     |
| Responsible AI Liaison (Sam Khozama / Mohan K. Samavedam) | §4 & §5 — defines *what* to test for: fairness, transparency, reliability, safety, bias mitigation criteria.           |
| Data Architecture Liaison (Sreeni Iyer / Aman Agarwal) | §8 — eval dataset governance; golden-set lineage; labeling standards; data-quality bindings.                            |
| Information Security Liaison                      | §7 & §11 — adversarial testing menu; penetration testing coordination; security gates in eval.                                          |
| Application / Business-Unit Representatives       | One per profile (Industrial / Customer / Corporate) — profile-specific eval criteria, business-SLA inputs, exception input. |
| Risk & Compliance Liaison                         | §6 — Model Risk Management framework consumption; audit-trail requirements.                                                  |
| Vendor / Partner Liaison                          | Eval tooling vendor input; penetration testing partner liaison (if external partner is chosen).                                         |

---

## Evaluation & Test Strategy Functions

The Evaluation & Test Strategy group will provide the following functions:

1. Evaluation Framework & Taxonomy
2. Metrics, Thresholds & SLAs
3. Evaluation Datasets & Tooling
4. Adversarial & Safety Testing
5. Production Monitoring, Drift & Continuous Improvement
6. Eval Gates & Lifecycle Integration

### Evaluation Framework & Taxonomy

The **Evaluation Framework & Taxonomy** function shall define and maintain the canonical map of what we evaluate, when, and against which dataset — **calibrated for agentic workflows**.

- **Four evaluation tracks** every AI solution conforms to:
  - **Offline (pre-deployment)** — golden-set accuracy, regression vs prior versions, holdout performance, calibration, agent-trace replay against canonical scenarios.
  - **Online (production)** — live-traffic quality sampling, drift detection, cost/latency monitoring, user-feedback signal, LLM-as-judge sampling, policy-check enforcement on every action.
  - **Adversarial (penetration testing)** — prompt injection, jailbreak resistance, data exfiltration probes, bias amplification, output-handling exploits, system-prompt extraction, **plus agent-specific vectors** (tool misuse, forbidden-action attempts, plan injection, reasoning hijack).
  - **Periodic Revalidation** — scheduled re-evaluation against current golden sets, model-drift remediation, dataset refresh cycles.

- **Agentic Evaluation Dimensions** (the dimensions that make agentic workflows different from single-turn LLM eval):
  - **Tool-selection correctness** — did the agent call the *right* tool (vector search vs DB query vs API call)? Were arguments well-formed?
  - **Reasoning step quality** — is each reasoning step (chain-of-thought, plan step) justified by available context?
  - **Step-sequence correctness** — are steps in the right order? (Critical for Plan-and-Execute and Manager-Worker patterns from §9.2.)
  - **Task-completion rate** — full completion vs partial completion vs failure; classified by failure mode.
  - **Trace traceability & replayability** — every step logged (inputs, outputs, latency, cost, policy verdict) and replayable for post-hoc review.

- **LLM Adaptation Strategy ↔ Eval Mapping** — evaluation focus shifts based on how the solution was built (the adaptation choice is owned by §9.2 / §10; the corresponding eval menu is owned by this group):
  - **Prompt engineering** — base LLM, no extra data. Eval emphasis: prompt robustness · output consistency · edge-case handling · regression on prompt changes.
  - **RAG (Retrieval-Augmented Generation)** — external / fresh knowledge via retrieval. Eval emphasis: **retrieval quality (Recall@K, Precision@K, MRR)** + generation grounding + context-coverage.
  - **Fine-tuning** — model adapted to domain style/behavior. Eval emphasis: regression vs base model · domain-pattern accuracy · training-data quality & contamination · structured-output correctness.
  - *Rule of thumb:* start with prompt engineering; if the model lacks knowledge → RAG; if the model lacks behavior/pattern consistency → fine-tune. Eval menu changes accordingly.

- **Per-profile eval emphasis** — the same taxonomy applies to all profiles, but the **weight** differs:
  - *Industrial / Operational* — safety, reliability, HITL effectiveness, tool-selection correctness, OT-context accuracy, forbidden-action prevention weighted highest.
  - *Customer / Commercial* — hallucination, groundedness, brand-safety, citation accuracy, IP-boundary respect weighted highest.
  - *Corporate / Productivity* — accuracy, employee-data privacy, productivity-lift measurement, acceptance rate weighted highest.

- **Per-pattern eval emphasis** — the agentic patterns from §9.2 each have signature evaluation needs:
  - **Grounded Q&A (RAG)** — retrieval quality + generation grounding + citation accuracy.
  - **Single-agent ReAct** — tool-selection correctness + reasoning step quality + completion rate.
  - **Plan-and-Execute** — plan correctness + step-sequence correctness + step-level completion + rollback safety.
  - **Manager-Worker multi-agent** — task decomposition quality + sub-task success rate + coordination overhead.
  - **Peer-to-peer handoff** — handoff fidelity (context preservation across agents) + role-adherence per specialist.

- **Maintenance cadence:** quarterly review; out-of-band review for major model or platform change.
- **Deliverable:** *Evaluation Taxonomy v1*, with the four-track map, agentic dimensions, adaptation-strategy mapping, per-profile weighting, and per-pattern eval signature.

### Metrics, Thresholds & SLAs

The **Metrics, Thresholds & SLAs** function shall maintain the catalog of measurements every AI solution is judged against — organized into **seven evaluation dimensions calibrated for agentic GenAI workflows**.

**Dimension 1 — Retrieval Quality (RAG systems)** — *Are we fetching the right context?*
- **Recall@K** — coverage of relevant documents in top-K results (e.g., Recall@5 = 3 relevant / 5 retrieved = 0.6).
- **Precision@K** — proportion of top-K results that are actually relevant.
- **MRR (Mean Reciprocal Rank)** — how high the *first* relevant result ranks; rewards retrieval that surfaces the right answer near the top.
- **Context coverage** — fraction of query intent satisfied by retrieved context.
- **Retrieval latency** — time spent in retrieval step.

**Dimension 2 — Generation Quality (LLM output)** — *Is the response correct, complete, structured?*
- **Exact match** — for dictionary / structured outputs (numbers, codes, IDs, JSON schemas).
- **F1 score** — token-level overlap between predicted and ground-truth answer (span-extraction, structured text).
- **Semantic similarity** — embedding-based comparison with ground truth where exact-match is too strict.
- **LLM-as-judge** — independent model (GPT / Claude) scoring outputs on the **five GenAI dimensions**: **relevance · completeness · groundedness · correctness · clarity**.
- **Schema-conformance rate** — for structured outputs, fraction that match required schema.

**Dimension 3 — Groundedness & Hallucination Detection**
- **Ground-truth comparison** — does every factual claim trace to retrieved context or known truth?
- **LLM-as-judge for hallucination** — independent model flagging unsupported claims.
- **Rule-based checks** — citation requirement, source-document presence, "no claim without source" rule.
- **Hallucination rate** — fraction of responses containing at least one unsupported factual claim.

**Dimension 4 — Agentic Behavior** — *Did the system take the right steps with the right tools?*
- **Tool-selection correctness** — did the agent call the *correct* tool (vector search vs DB query vs API call)? Wrong-tool rate per profile.
- **Tool-argument correctness** — are tool arguments well-formed, complete, and non-injected?
- **Reasoning step quality** — is each reasoning step justified by available context? (LLM-as-judge over chain-of-thought.)
- **Step-sequence correctness** — are steps executed in the right order? (Critical for Plan-and-Execute and Manager-Worker patterns.)
- **Task-completion rate** — full completion / partial completion / failed; classified by failure mode (planning failure · tool failure · reasoning failure · external system failure).
- **Trace replayability** — every step logged with inputs, outputs, latency, cost, policy-check verdict; trace is replayable end-to-end for post-hoc review.

**Dimension 5 — Business Validation (Human-in-the-Loop)** — *Is the solution actually useful?*
- **Manual-effort reduction** — measured time saved per task vs the pre-AI baseline.
- **User satisfaction** — explicit ratings (thumbs / 1-5), escalation rate, correction rate.
- **Acceptance rate** — fraction of agent suggestions adopted by the user (e.g., "engineers accepted the similarity suggestion 78% of the time").
- **Task throughput** — tasks completed per user per period, with vs without AI.
- **Cost-of-mistake avoidance** — averted-incident value (Profile 1 safety-critical use cases).

**Dimension 6 — Operational (Latency & Cost)**
- **Latency** — time per request: p50 / p95 / p99. End-to-end and per-agent-step.
- **Token usage** — input + output tokens per task. Trended per profile per month.
- **API-call count** — number of LLM / tool / retrieval calls per task (agentic workflows can fan out badly).
- **Cost per task** — dollars per task; rolled up to cost per profile per month; per-customer for Profile 2.
- **Throughput** — tasks served per second; error rate.

**Dimension 7 — Safety & Policy** — *Is every agent action policy-compliant?*
- **PII / confidential-data exposure rate** — outbound disclosure events; zero-tolerance threshold for Profile 2.
- **Toxicity / harmful-content rate** — toxicity classifier on outputs.
- **Restricted-content rate** — domain-rule violations (industry-specific restrictions, regulatory).
- **Brand-safety score** — Profile 2 customer-facing surfaces.
- **Forbidden-action attempts** — agent attempting to send emails, delete records, approve payments, execute financial transactions, modify production data, or take other policy-restricted actions **without explicit HITL approval**. Tracked as a leading indicator of agent-misalignment risk.
- **Policy-check pass rate** — fraction of agent actions that pass every real-time policy validator before execution.

**Per-profile thresholds.** Every metric in every dimension has a profile-specific threshold. Examples:
- Hallucination rate: Profile 2 (customer) stricter than Profile 3 (internal copilot).
- Tool-selection correctness: Profile 1 (industrial safety-critical) is the strictest.
- Forbidden-action attempt rate: zero tolerance across all profiles; any breach triggers immediate review.

**SLA bindings.** Thresholds become contractual SLAs for production AI solutions — breach triggers alert routing and exception workflow.

**Deliverable:** *Metrics, Thresholds & SLAs Catalog v1*, with all seven dimensions, per-metric definition, calculation method, profile-specific threshold, alert routing, and exception process.

### Evaluation Datasets & Tooling

The **Evaluation Datasets & Tooling** function shall define the standards, governance, and approved-tool list for all evaluation data and evaluation execution.

- **Dataset categories:**
  - **Golden sets** — curated, labeled, profile- and pattern-specific. The canonical "must pass" set for any deployment.
  - **Regression sets** — historical examples that previously failed; ensure fixes stay fixed.
  - **Holdout sets** — never seen during training / fine-tuning; sampled per evaluation.
  - **Synthetic sets** — generated for edge-case coverage where natural data is scarce.
  - **Adversarial sets** — pen-test-curated prompts and inputs.
- **Dataset governance:** versioning (semantic), lineage (per Section 8 standards), labeling protocols (multi-annotator with adjudication for safety-critical sets), refresh cadence.
- **Eval-tooling slot** (vendor-pluggable per §9 catalog convention):
  - **[OPEN]** Default candidate: Azure AI Studio Evaluations + PromptFlow.
  - **[OPEN]** Alternatives: DeepEval (open-source), Ragas (RAG-specific), or hybrid.
  - Selection criterion: integration with §9 eval-hook contract, profile coverage, cost predictability, ecosystem.
- **Integration:** all tools emit results in the standard eval-hook format defined with §9, so results are interpretable across tools and over time.
- **Deliverable:** *Evaluation Datasets & Tooling Standards v1*, with the dataset taxonomy, governance, and selected eval-harness slot.

### Adversarial & Safety Testing

The **Adversarial & Safety Testing** function shall operate the penetration testing discipline that pressure-tests every production-bound AI solution.

- **Penetration Testing scope — classical LLM vectors (derived from §7 Threat Taxonomy):**
  - Prompt injection (direct and indirect)
  - Jailbreak attempts and persona override
  - Sensitive information disclosure / data exfiltration
  - Data and model poisoning probes
  - Improper output handling
  - System prompt exposure
  - Vector / embedding vulnerabilities
  - Overreliance / hallucination amplification
  - Bias amplification under adversarial input
- **Penetration Testing scope — agent-specific adversarial vectors (the new attack surface from agentic workflows):**
  - **Tool misuse** — coercing the agent into calling tools it shouldn't, or calling them with the wrong arguments
  - **Forbidden-action attempts** — adversarial input designed to get the agent to send emails, delete records, approve payments, modify production data, or execute other policy-restricted actions
  - **Multi-step plan injection** — adversarial input that poisons the planning step (Plan-and-Execute pattern) — looks benign at step 1, malicious at step 4
  - **Reasoning hijack** — steering the agent's reasoning chain off the legitimate task
  - **Tool-argument injection** — malicious payloads injected into tool arguments that propagate downstream (e.g., SQL-injection-like attacks via DB-query tools)
  - **Handoff exploitation** — for multi-agent patterns, exploiting context transfer between agents
- **Engagement model:** every Profile 2 (customer-facing) solution is pen-tested before production; Profile 1 (industrial) solutions are pen-tested when they cross the HITL boundary into autonomous write actions; Profile 3 (corporate) solutions are pen-tested by category.
- **[OPEN]** **Penetration Testing partner decision:** internal-only team vs hybrid with external partner. Recommend hybrid for Profile 2 customer-facing surfaces.
- **Findings flow:** penetration testing findings feed into the metrics catalog (new tests added), the guardrail policies in §9 (L3 guardrail updates), and the lifecycle exception process (§10).
- **Deliverable:** *Adversarial & Safety Testing Playbook v1*, with the attack menu, engagement model, partner decisions, and findings workflow.

### Production Monitoring, Drift & Continuous Improvement

The **Production Monitoring, Drift & Continuous Improvement** function shall define how AI solutions are watched in production and how their performance is recovered when it degrades.

- **Online evaluation methodology (continuous, every request):**
  - Live-traffic sampling rate per profile (Profile 2 higher than Profile 1 for low-risk write actions).
  - Shadow evaluation against newer model versions before promotion.
  - **User-feedback signal capture** — thumbs-up/down, escalation rate, correction rate.
  - **LLM-as-judge** — independent model sampling production responses for groundedness, hallucination, brand-safety.
  - **Rule-based validators** — citation requirements, output-schema conformance, no-claim-without-source.
  - **HITL review** — human review on a sampled fraction of high-risk actions (especially Profile 1 write actions and Profile 2 customer-impact actions).
  - **Trace logs per step** — every agent step (reasoning, tool call, retrieval, generation) logged with inputs, outputs, latency, cost, policy-check verdict — replayable end-to-end for incident review.

- **Policy checks (online, every agent action — *before* execution, not after):**
  - **PII / confidential-data outbound** — blocked at the boundary if disclosed.
  - **Toxic / unsafe / restricted content** — output blocked, alert raised.
  - **Domain-rule conformance** — industry-specific rules (e.g., Profile 1 plant ops cannot recommend procedures that violate process-safety constraints).
  - **Forbidden-action prevention** — agents are *physically prevented* from sending emails, deleting records, approving payments, modifying production data, or executing other policy-restricted actions **without explicit HITL approval**. This is not a detection rule; it is an enforcement rule. Attempts are logged as leading indicators of misalignment.

- **Drift detection:**
  - **Input drift** — population stability index over rolling windows.
  - **Output drift** — response-distribution shift.
  - **Embedding drift** — semantic shift in retrieved context.
  - **Performance drift** — rolling quality metrics vs baseline.
  - **Tool-use drift** — changes in agent tool-selection patterns over time (e.g., agent suddenly favoring a different tool — may indicate prompt-injection campaign or model regression).
- **Alert routing:** drift events route to on-call AI engineers; SLA breaches route to the application owner + Eval & Test group; safety breaches route to Information Security + Responsible AI.
- **Continuous improvement loop:**
  - Production failures → regression set growth.
  - User feedback → golden-set refinement.
  - Drift events → model re-evaluation and (if needed) retraining or prompt updates.
  - Quarterly model-portfolio review with §10 (AI Lifecycle).
- **Deliverable:** *Production Monitoring & Drift Detection Methodology v1*, with per-profile sampling, drift thresholds, alert routing, and the continuous-improvement workflow.

### Eval Gates & Lifecycle Integration

The **Eval Gates & Lifecycle Integration** function shall define the governance checkpoints where evaluation results decide whether a solution advances, holds, or rolls back. This function is **co-owned with §10 (AI Development Lifecycle)**: §10 owns the *pipeline execution* of the gates; Evaluation & Test Strategy owns the *gate criteria* (what passes, what fails, what triggers exception review).

- **Gate 1 — Pilot → Pre-deployment:** offline eval against golden set; minimum bias/safety thresholds; Pen Test Tier 1 (basic).
- **Gate 2 — Pre-deployment → Production:** full offline eval; Pen Test Tier 2 (profile-specific); cost & latency SLA confirmation; final HITL workflow check.
- **Gate 3 — Production health (continuous):** drift within thresholds; user-feedback signal within bounds; SLA compliance over rolling window.
- **Gate 4 — Periodic Revalidation (quarterly or on trigger):** re-evaluate against current golden set; refresh dataset where applicable; re-run penetration testing for new attacks; model-version reassessment.
- **Exception process:** when a solution fails a gate, the application owner can submit an exception with documented mitigation, security review sign-off, and approval by the Eval & Test group plus the relevant IT Director.
- **Deliverable:** *Eval Gates & Lifecycle Integration v1*, with gate criteria, exception process, and §10 pipeline interface specification.

---

## Evaluation & Test Strategy Meetings

At each meeting the Evaluation & Test Strategy group, co-led by **Sarath Mandadi and Aman Agarwal**, will use an agenda outline containing at a minimum the following:

- Roll Call
- Review Agenda
- Minutes from last meeting
- Review of actions arising from previous Evaluation & Test Strategy meetings
- New items arising since the last meeting
- Active gate exception submissions
- Production-drift / SLA-breach incidents requiring eval response
- Cross-section dependencies (Technical Architecture, AI Lifecycle, Responsible AI, Data Architecture, Information Security)
- Plans, date and location for next meeting

Initial meeting cadence: **weekly** during framework build-out (Jun – Oct 2026); **bi-weekly** thereafter once v1 strategy is published; ad-hoc as needed for high-priority gate exceptions or production incidents.

---

## Related Documents

The following documents are related to this document and should be reviewed in parallel:

- *AI Framework — Section 4: Principles for Responsible Artificial Intelligence*, Celanese
- *AI Framework — Section 5: Responsible AI Implementation and Ethics*, Celanese
- *AI Framework — Section 6: Risk Management and Compliance*, Celanese
- *AI Framework — Section 7: AI Threat Landscape and Risk Taxonomy*, Celanese
- *AI Framework — Section 8: Data Management and Governance*, Celanese, Data Architecture working group (Sreeni Iyer, Aman Agarwal)
- *AI Framework — Section 9: Technology Architecture*, Celanese, Technical Architecture working group (Sarath Mandadi, Mohan Krishna Samavedam)
- *AI Framework — Section 10: AI Development Lifecycle (MLOps / LLMOps)*, Celanese, AI Lifecycle working group (Nagesh Cheripally, Sreeni Iyer)
- *AI Framework — Section 11: Security and Cyber Risk Management*, Celanese, Information Security
- *AI Framework — Section 12: Generative AI Governance Guidelines*, Celanese
- *Section 9 — Technology Architecture Charter*, Celanese, Technical Architecture working group — for the eval-hook contract definition

---

> **End of Charter — v0.1 Draft**
