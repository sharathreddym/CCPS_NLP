# Chemille → X1 Account Manager — Final Report

**Goal:** assign an X1 Account Manager to every Chemille portal user, in two phases:
- **Phase 1** — deterministic lookup (Chemille → SFDC → SAP) for users we can trace.
- **Phase 2** — one combined prediction model (a 4-tier waterfall) for the rest.

---

## 1. Headline result

Of **4,854 external Chemille users** (after removing 1,447 internal `@celanese.com`):

| | Users | % |
|---|---:|---:|
| ✅ Confirmed (exact lookup + enrichment) | **1,510** | 31% |
| 🔵 Predicted (model) | 3,344 | 69% |
| **Auto-usable** (confirmed + high-confidence) | **3,049** | **63%** |
| 👁️ Flagged for human review | **1,805** | **37%** |

Blended held-out accuracy of the prediction model: **~82%**.

---

## 2. Full breakdown — every user accounted for

```
6,301  RAW Chemille users
│
├── 1,447  internal @celanese.com            → OUT OF SCOPE (not customers)
│
└── 4,854  EXTERNAL customers (100%)
    │
    ├── 783   PHASE 1 — exact email match           [confirmed, ground truth]
    │
    └── 4,071 PHASE 2 — combined model (waterfall):
        ├── 727   Tier 1  Enrichment (email domain)    97.7%   ┐ confirmed
        │                                                        ├─ 1,510 total confirmed
        ├── 440   Tier 2  Domain lookup (783)           81.8%   ┐ strong
        ├── 492   Tier 3  Account-name similarity       80.6%   ┘ (932)
        └── 2,412 Tier 4  Text similarity (fallback)    37.9%   → 607 accept / 1,805 review
```

### Stage by stage (how "unknown" shrinks)

| Stage | Confirmed / handled | Still unknown |
|---|---:|---:|
| Start (external users) | 0 | 4,854 |
| After Phase 1 (exact email) | 783 | 4,071 |
| After Tier 1 enrichment | **1,510** | 3,344 |
| After Tiers 2 + 3 | **2,442** | 2,412 |
| After Tier 4 (fallback) | 4,854 | 0 |

### Final disposition by quality

| Group | Users | % | Quality |
|---|---:|---:|---|
| Confirmed — Phase 1 + Enrichment | **1,510** | 31.1% | 96–100% |
| Strong predictions — Tier 2 + Tier 3 | **932** | 19.2% | ~81% |
| Fallback predictions — Tier 4 | **2,412** | 49.7% | ~38% (mostly review) |

---

## 3. The combined prediction model (one model, 4-tier waterfall)

This is a **single model**. Given a Chemille user, it tries the strongest method
first and **stops at the first tier that is confident**. If a tier isn't
confident, it falls through to the next.

| Tier | Method | Match key | Built from | Confident when | Held-out acc. |
|---|---|---|---|---|---:|
| 1 | Enrichment | email **domain** | full SFDC Contacts → Account → SAP | domain in directory (purity ≥ .6) | 97.7% |
| 2 | Domain lookup | email **domain** | the 783 labels | domain vote share ≥ .8 | 81.8% |
| 3 | Account-name similarity | **company name** | SFDC Accounts + SAP | name match ≥ .85 **and** purity ≥ .6 | 80.6% |
| 4 | Text similarity (fallback) | company+domain+country **text** | the 783 labels | always (last resort) | 37.9% |

### What happens for a NEW user

```
new user (email, company, country)
   │  take email_domain (text after @)
   ▼
Tier 1: domain in the enrichment directory?          ── yes → return X1, STOP
   │ no
Tier 2: domain among the 783 labels (vote ≥ .8)?     ── yes → return X1, STOP
   │ no
Tier 3: company NAME matches an SFDC account
        (match ≥ .85 & purity ≥ .6)?                  ── yes → return X1, STOP
   │ no
Tier 4: nearest known company by TEXT → return X1
        (flagged for review if the match is weak)
```

Note the match key **changes by tier**: Tiers 1–2 use the **email domain**;
Tier 3 switches to the **company name**; Tier 4 uses the combined **text**. The
domain is the strongest signal when available; company name / text are the
fallbacks when the domain is unknown.

---

## 4. How to use it

### A) The ready-made answers — `chemille_users_x1_FINAL_all.csv`
One row per Chemille user with their assigned manager:

| Column | Meaning |
|---|---|
| `chemille_email`, `company`, `country`, `email_domain` | the user |
| `X1_Account_Manager` | the assigned manager |
| `source` | how it was found (Phase 1 / Enrichment / which prediction tier) |
| `confidence` | 0–1 |
| `needs_review` | `True` → please check before using |

**Recommended:** apply directly where `needs_review = False` (3,049 users); send
the `needs_review = True` rows (1,805) to the commercial team for a quick check.

### B) Predict a NEW user — `final_model_pipeline.ipynb` / `.py`
Run the notebook, then call the helper in the last cell:

```python
predict_final_one(company="Parker Hannifin", email="newbuyer@parker.com", country="Japan")
# -> TIMOTHY BUTURLA | method=2_domain_lookup_783 | confidence=1.0 | needs_review=False
```

The model rebuilds its reference data from the four source files
(`Chemille-User-Data.csv`, `SFDC-Contact-Data.csv`, `SFDC-Accounts-Data.csv`,
`SAP-Data.csv`), so those must sit alongside the notebook when running it.

---

## 5. Folder contents

| File | What it is |
|---|---|
| `FINAL-REPORT.md` | this document |
| `chemille_users_x1_FINAL_all.csv` | **the result** — all 4,854 users + their X1 |
| `final_model_pipeline.ipynb` / `.py` | **the model** — the 4-tier waterfall + `predict_final_one()` |
| `approach_diagram.png` | the waterfall, visualised |

---

### One-line summary
> From 6,301 raw Chemille users, 1,447 are internal (out of scope), leaving 4,854
> external customers. A two-phase approach — deterministic lookup (Phase 1) plus a
> single 4-tier prediction model (Phase 2) — assigns an X1 to every user, with
> **3,049 (63%) auto-usable** and **1,805 (37%) routed to human review**, at ~82%
> blended accuracy.
