# Chemille Users — Full Breakdown (Phase 1 → Phase 2)

How every Chemille user is accounted for, from the raw file through Phase 1
(deterministic mapping) and Phase 2 (enrichment + prediction). All numbers are
from the current data extract.

---

## 0. Starting point — the raw file

**`Chemille-User-Data.csv` = 6,301 rows**

| Removed before we start | Users | Reason |
|---|---:|---|
| Internal `@celanese.com` employees | **1,447** | Celanese staff, not external customers — out of scope |
| Blank emails | 0 | none in this extract |
| Duplicate emails | 0 | none in this extract |

**→ Working universe = 4,854 distinct EXTERNAL customers (100%).**
Everything below is a breakdown of these 4,854.

---

## 1. PHASE 1 — deterministic mapping (exact email)

Chain: `Chemille email → SFDC Contact → SFDC Account → SAP → X1`
(with DQ #1 ambiguous-email removal and DQ #2 invalid-X1 removal).

| Result of Phase 1 | Users | % of 4,854 |
|---|---:|---:|
| ✅ **LABELED** (got a confirmed X1) | **783** | 16.1% |
| ⬜ **UNLABELED** (exact email not in SFDC) | **4,071** | 83.9% |

- The **783** are ground-truth labels (used to train Phase 2).
- The **4,071** go into Phase 2.

---

## 2. PHASE 2 — enrichment + prediction (the 4,071)

The 4,071 unlabeled users pass through a **waterfall**: each user is assigned by
the FIRST method that is confident, strongest method first.

| Tier | Method | Match key | Built from | Users | Held-out accuracy |
|---|---|---|---|---:|---:|
| **1** | Enrichment | email **domain** | full SFDC Contacts → Account → SAP | **727** | **97.7%** |
| **2** | Domain lookup | email **domain** | the 783 labels (majority vote) | **440** | 81.8% |
| **3** | Account-name similarity | **company name** | SFDC Accounts + SAP (with purity) | **492** | 80.6% |
| **4** | Text similarity (fallback) | company+domain+country **text** | the 783 labels (nearest) | **2,412** | 37.9% |
| | | | **Total** | **4,071** | **82.1% blended** |

*(Tiers 1 & 3 read SFDC/SAP that contains the companies, so their accuracy is
slightly optimistic — same caveat as the enrichment validation.)*

---

## 3. Labeled vs Unlabeled — how it shrinks stage by stage

| Stage | Confirmed / handled | Still "unknown" |
|---|---:|---:|
| Start (external users) | 0 | 4,854 |
| After **Phase 1** (exact email) | 783 | 4,071 |
| After **Tier 1 enrichment** | 783 + 727 = **1,510** | 3,344 |
| After **Tier 2 + Tier 3** | 1,510 + 932 = **2,442** | 2,412 |
| After **Tier 4** (fallback) | all 4,854 covered | 0 |

---

## 4. Final disposition of all 4,854 users (by quality)

| Group | Users | % | Quality |
|---|---:|---:|---|
| **Confirmed** — Phase 1 + Enrichment | **1,510** | 31.1% | deterministic / 96–98% |
| **Strong predictions** — Tier 2 + Tier 3 | **932** | 19.2% | ~81% |
| **Fallback predictions** — Tier 4 | **2,412** | 49.7% | ~38% (mostly review) |
| **Total** | **4,854** | 100% | |

---

## 5. Action split — auto-accept vs human review

| Disposition | Users | % of 4,854 | Made of |
|---|---:|---:|---|
| ✅ **Auto-accept** | **3,049** | 62.8% | 783 (Phase 1) + 1,659 (Tiers 1–3) + 607 (strong Tier-4) |
| 👁️ **Needs human review** | **1,805** | 37.2% | weak Tier-4 (text similarity < 0.50) |

> On the 4,071 Phase-2 users alone: **2,266 auto-accept, 1,805 review.**

---

## 6. The whole picture in one tree

```
6,301  RAW Chemille users
│
├── 1,447  internal @celanese.com            → OUT OF SCOPE
│
└── 4,854  EXTERNAL customers (100%)
    │
    ├── 783   PHASE 1 — exact email match           [confirmed, ground truth]
    │
    └── 4,071 PHASE 2 (waterfall):
        ├── 727   Tier 1  Enrichment (domain)         97.7%   ┐ confirmed
        │                                                       ├─ 1,510 total confirmed
        ├── 440   Tier 2  Domain lookup (783)          81.8%   ┐ strong
        ├── 492   Tier 3  Account-name similarity      80.6%   ┘ (932)
        └── 2,412 Tier 4  Text similarity fallback     37.9%   → 607 accept / 1,805 review
```

---

## 7. One-line summary

> From **6,301** raw Chemille users, **1,447** are internal (out of scope), leaving
> **4,854** external customers. Phase 1 confirmed **783**; Phase 2 enrichment added
> **727** near-certain (**1,510 confirmed**), then predicted the rest — leaving
> **3,049 users (63%) auto-accepted** and **1,805 (37%) for human review**, at an
> honest **82% blended accuracy**.

---

### Source files
- Master predictions: `chemille_user_x1_FINAL_predictions.csv`
- Pipeline: `phase2_combined_pipeline.py` / `.ipynb`
