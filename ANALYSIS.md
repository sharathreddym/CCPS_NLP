# Chemille User → X1 Account Manager — Analysis & Documentation

**Goal:** for every Chemille portal user (an external customer), determine their
**X1 Account Manager** so the mapping can drive account alignment and
account-manager incentives. No single system stores this link, so we build it in
two phases:

- **Phase 1 (done):** deterministic joins across Chemille → SFDC → SAP to produce
  a *known-correct* labeled dataset.
- **Phase 2 (next):** a machine-learning model that predicts X1 for the majority
  of users that joins cannot reach.

> Data refreshed 2026-06-14. The SFDC files grew substantially (Contacts → 119
> columns / ~245 MB, Accounts → 102 columns / ~148 MB, 104,764 account rows).
> Pipelines now read only the columns they need (`usecols`) for memory & speed.

---

## 1. Data landscape

| File | Rows | Cols | Role |
|------|-----:|-----:|------|
| `Chemille-User-Data.csv` | 6,301 | 5 | Portal users (Email, firstName, lastName, country, company) |
| `SFDC-Contact-Data.csv` | 145,870 | 119 | **People** at customer companies (the email bridge) |
| `SFDC-Accounts-Data.csv` | 104,764 | 102 | **Companies** = the customers (hold the SAP id + region/industry/type) |
| `SAP-Data.csv` | 82,969 | 9 | SAP customers + X1/X3/X4/X9 managers (**source of truth**) |

**CRM data model:** in Salesforce an **Account = the customer company** and a
**Contact = an individual person** at that company. X1 is assigned to the
*company*, not the person — so the chain must go *person → their company → who
manages that company*.

---

## 2. The mapping chain

```
CHEMILLE_USER.Email
      │  exact email match (normalized)
      ▼
SFDC_CONTACT.AccountId
      │  AccountId  ==  Account.Id   (case-sensitive 18-char)
      ▼
SFDC_ACCOUNT.SAP_Customer_Master_ID__c
      │  normalized SAP id
      ▼
SAP.[SAP Customer ID]
      ▼
X1 (+ X3 / X4 / X9)  Account Manager
```

---

## 3. Data Quality Controls (planning doc, page 2)

| # | Control | Implementation | Effect (current data) |
|---|---------|----------------|----------------------|
| **1** | Remove emails linked to **multiple** SFDC accounts | group contacts by email; if it maps to >1 distinct `AccountId`, **drop all rows** for that email | dropped **4,122** ambiguous emails |
| **2** | Exclude **invalid/system** assignments | `X1` is free-text; drop case-insensitive matches of `not assigned / unassigned / outdated / transferred` | dropped **13,245** SAP rows |
| **3** | **Normalize identifiers** across systems | identical cleaning on **both sides** of every join key | see §4 |

### DQ #2 — the "invalid" values are literal strings inside the `X1` column
| X1 value | SAP rows |
|----------|---------:|
| Not assigned | 5,953 |
| TRANSFERRED TO EU CHANNEL PARTNER | 3,015 |
| UNASSIGNED ACCOUNT REP | 1,877 |
| TRANSFERRED TO US CHANNEL PARTNER | 1,568 |
| OUTDATED UNASSIGNED | 829 |
| UNASSIGNED SALES REP ASIA | 2 |
| UNASSIGNED CSR | 1 |
| **Total excluded** | **13,245** |

Substring matching (not an exact list) catches long-tail variants an exact list would miss.

---

## 4. Normalization map (DQ #3)

Normalize **only join keys**, **identically on both sides**:

| Join key | Files (both sides) | Treatment |
|----------|--------------------|-----------|
| `email` | Chemille `Email` + Contact `Email` | trim → lowercase → strip leading `*` (bounce markers) |
| `account_id` | Contact `AccountId` + Account `Id` | **trim only — NOT lowercased** (case-sensitive 18-char Ids) |
| `sap_id` | Account `SAP_Customer_Master_ID__c` + SAP `SAP Customer ID` | trim → drop trailing `.0` → strip leading zeros |

`X1` is `.strip()`-ed but **casing preserved** (`MARCO SECCHI`, not `marco secchi`).
Whitespace audit on `account_id`: 0 rows changed by `.strip()` — kept as defensive insurance against future refreshes.

---

## 5. Phase-1 result — the funnel

From external Chemille users (after dropping 1,447 internal `@celanese.com`):

| Stage | Users | % of external |
|-------|------:|--------------:|
| Chemille external users | 4,854 | 100.0% |
| → matched an SFDC contact (email) | 1,261 | 26.0% |
| → contact linked to an Account | 867 | 17.9% |
| → account had a SAP Customer ID | 840 | 17.3% |
| → **resolved to a VALID X1 (LABEL)** | **783** | **16.1%** |

- **783 labeled users** across **193 distinct X1 managers** → the Phase-1 training set.
- **4,071 users (84%)** remain unlabeled → the Phase-2 (ML) targets.

---

## 6. NEW: what actually determines the X1 manager? (Cramér's V vs X1)

Measured on the labeled set. Cramér's V ranges 0 (no association) → 1 (perfect).
**Availability** = is the attribute known for an *unlabeled* user (a valid model
input) or only after matching (would be leakage)?

| Feature | Cramér's V vs X1 | Availability |
|---------|-----------------:|--------------|
| `account_region` | **0.86** | company-context (leakage) |
| `account_type` | 0.77 | company-context (leakage) |
| **`email_domain`** | **0.69** | **user-available ✅** |
| `account_country` (BillingCountry) | 0.60 | company-context |
| `account_industry` | 0.60 | company-context |
| `sap_salesorg` | 0.57 | company-context |
| **`country`** | **0.53** | **user-available ✅** |
| **`company`** | **0.50** | **user-available ✅** |
| `sap_subseg` | n/a (1 level) | — |

**Conclusions:**
1. X1 is essentially **company- and region-determined** (region 0.86, type 0.77) —
   confirming X1 is an account-level attribute, not a per-person one.
2. Of the attributes we can actually use for unlabeled users, **`email_domain`
   (0.69) is by far the strongest** predictor, followed by `country` (0.53) and
   `company` (0.50). This directly validates the Phase-2 design below.
3. The strongest drivers (region/type/industry/salesorg) are **not available for
   unlabeled users** — using them as inputs would be data leakage. `country` is
   the best *user-side* proxy for region.

---

## 7. Key observations & conclusions

1. **The bottleneck is the email match (hop 1).** Only **26%** of users match an
   SFDC contact — that single step caps Phase-1 coverage. The SAP side is
   healthy (100,856 of 104,764 accounts carry a SAP id). **To raise coverage,
   improve the email match, not the SAP join.**
2. **Extract quality matters.** An earlier (smaller) extract produced **0**
   end-to-end labels because contact↔account ids barely overlapped. Always
   sanity-check key overlap before trusting a run.
3. **A direct Chemille-email → SAP shortcut is not viable** (users missing in
   SAP, inconsistent company names, many-to-many explosion). The SFDC bridge is
   necessary.
4. **Labels are sparse & long-tailed:** 783 labels / 193 managers ≈ **4 per
   manager**, with many managers having a single user. This is hard for a 193-way
   classifier → see Phase-2 mitigations.
5. **`X1` is dirty free-text** mixing names with 13,245 placeholder rows — DQ #2
   is essential, not cosmetic.
6. **Governance satisfied:** SAP = source of truth, SFDC = label generation,
   model = prediction only; every label is fully traceable
   (`chemille_email → sfdc_account_id → sap_customer_id → X1`).

---

## 8. Phase 2 — design (evidence-based)

**Task:** predict `X1_Account_Manager` for the 4,071 unlabeled users.
**Inputs (user-available only):** `email_domain`, `country`, `company`.
**Training data:** the 783 labeled rows.

Recommended tiered approach (matches the doc's confidence + human-in-the-loop):

1. **Tier 1 — deterministic domain lookup.** From labels, build
   `email_domain → X1` (majority vote). Predict for unlabeled users whose domain
   is already known. *High precision, fully explainable.* Justified by the 0.69
   association — domain almost determines the manager.
2. **Tier 2 — similarity / nearest-neighbour** (TF-IDF or embeddings on
   `company` + `domain`, optionally `country`) for users Tier 1 misses. The
   similarity score doubles as a **confidence** value.
3. **Tier 3 — fallback classifier** (e.g. gradient boosting / logistic
   regression on one-hot domain + country + TF-IDF company) only if needed.

**Mitigations for sparsity (4 labels/manager):**
- Recover more labels by lifting the 26% email match (each match = a free label).
- Consider a **coarser target** first (predict `region`/`segment` or a company
  cluster, then resolve the individual manager) — easier to learn, still useful.

**Evaluation:** hold out part of the 783 known users; report accuracy + per-class
coverage. Apply a **confidence threshold**: auto-accept high-confidence
predictions, route the rest to a human (wrong-but-confident is costly because
this feeds incentives).

---

## 9. Deliverables in this folder

| File | What it is |
|------|------------|
| `phase1_build_training_dataset.py` / `.ipynb` | Commented Phase-1 pipeline (memory-safe `usecols`); outputs the labeled dataset |
| `chemille_user_x1_training_dataset.csv` | **Output:** 783 labeled rows (features + company-context + labels) |
| `eda_chemille_x1.py` / `.ipynb` | EDA: funnel, distributions, class imbalance, **feature→X1 ranking**, reusable Cramér's V heatmap |
| `ANALYSIS.md` | This document |
| `_convert_to_ipynb.py` | Helper: `python _convert_to_ipynb.py file.py file.ipynb` |

**Output columns:** Chemille features (`chemille_email, firstName, lastName,
country, company, email_domain`) → **valid model inputs**; company context
(`sfdc_account_*`, `sap_customer_id`, `sap_sales_org`) → **analysis only, not
inputs**; labels (`X1_Account_Manager, X3_Name, X4_Name, X9_Name`).

---

## 10. Reusing the EDA for new columns

1. Add the column to the relevant `*_USECOLS` list (EDA section 0).
2. Carry it through the merges (section 5) so it lands on `df`.
3. Add it to `CANDIDATES` (section 11) / `assoc_cols` (section 12), flagging
   whether it is user-available (valid input) or company-context (leakage).
4. Re-run — the **feature→X1 ranking** and **association heatmap** immediately
   show how strongly the new column relates to X1, i.e. whether it helps Phase 2.

---

## 11. Recommended next steps

1. **Lift the 26% email match** — investigate non-matches (personal vs corporate
   domains, typos, absent contacts). Highest-leverage improvement.
2. **Build Phase 2 Tier-1 + Tier-2 prototype** on `chemille_user_x1_training_dataset.csv`
   with held-out evaluation and confidence thresholds.
3. **Decide target granularity** (individual manager vs region/segment) based on
   measured accuracy given ~4 labels/manager.
