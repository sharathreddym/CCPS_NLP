# Data Cleaning & Filtering Report — Chemille → X1 Mapping (Phase 1)

This document explains, file by file, **how many records we start with, how many
we remove, and exactly why** — with real examples from the data. It also lists
the **join columns** and the columns we drop (and the reason).

*Data as of 2026-06-14.*

---

## TL;DR — the Chemille user funnel

| Stage | Users | Removed here | Why |
|-------|------:|-------------:|-----|
| Raw Chemille rows | 6,301 | — | starting point |
| Drop blank email | 6,301 | 0 | no empty emails in this extract |
| Drop internal `@celanese.com` | 4,854 | **1,447** | internal employees, not external customers |
| Deduplicate by email | 4,854 | 0 | no duplicate emails |
| **Distinct external users** | **4,854** | | **this is our working universe** |
| → **Resolved to a valid X1 (LABEL)** | **783** | | success — used to train Phase 2 |
| → Could not be mapped | 4,071 | | reasons broken down in §5 |

**Bottom line:** of 4,854 external users, **783 (16.1%) got a confirmed X1
manager**; **4,071 (83.9%) could not be mapped** — overwhelmingly because their
email simply isn’t in Salesforce (see §5).

---

## The join path (which columns connect the four files)

```
CHEMILLE_USER.Email ──┐
                      │ normalized exact match
SFDC_CONTACT.Email ───┘
SFDC_CONTACT.AccountId ──┐
                         │ exact match (case-sensitive 18-char Id)
SFDC_ACCOUNT.Id ─────────┘
SFDC_ACCOUNT.SAP_Customer_Master_ID__c ──┐
                                         │ normalized match
SAP.[SAP Customer ID] ───────────────────┘
SAP → X1 / X3 / X4 / X9   (the answer)
```

Normalization applied to **both sides** of each key:
- **Email:** trim → lowercase → strip leading `*` (bounce markers).
- **Account Id:** trim only (case-sensitive — must NOT be lowercased).
- **SAP Id:** trim → drop trailing `.0` → strip leading zeros.

---

## 1. Chemille Users (`Chemille-User-Data.csv`)

**Raw: 6,301 rows · 5 columns** (`Email, firstName, lastName, country, company`).
All 5 columns are kept (they are the model features). Cleaning steps:

| # | Step | Removed | Reason | Example removed |
|---|------|--------:|--------|-----------------|
| A | Drop blank/empty email | 0 | no email = no possible join key | — |
| B | Drop internal `@celanese.com` | **1,447** | these are Celanese employees, not external customers | `aaron.hu@celanese.com`, `abdul.jangda@celanese.com` |
| C | Deduplicate by normalized email | 0 | one row per user | — |

**Result: 4,854 distinct external users.**

---

## 2. SFDC Contacts (`SFDC-Contact-Data.csv`) — the email→account bridge

**Raw: 145,870 rows · 119 columns.**

### Columns: we keep 2 of 119
| Kept (used) | Why |
|-------------|-----|
| `Email` | join key to Chemille users |
| `AccountId` | join key to SFDC accounts (the company) |

**Dropped: 117 columns** — none are needed to map a person to their company, and
most are marketing/enrichment noise that is largely empty. Examples of dropped
columns: `mkto_si__*` / `mkto71_*` (Marketo lead-scoring), `DOZISF__ZoomInfo_*`
(ZoomInfo enrichment), `Engagement_1..5__c`, `PhotoUrl`, `Fax`, `EmailBounced*`,
`SystemModstamp - Year/Quarter/Month/Day`, etc. *(Reason: not join keys, not
required for Phase 1, high missingness.)*

### Row cleaning & filtering
| # | Step | Rows removed | Reason | Example |
|---|------|-------------:|--------|---------|
| 1 | Normalize email (strip `*`) | 0 rows (6 emails fixed) | bounce-marker prefix would break the match | raw `*david.price@selecteng.ca` → `david.price@selecteng.ca` |
| 2 | Drop blank email/account | 2 | unusable as a bridge | — |
| 3 | **DQ #1 — remove ambiguous emails** | **8,884** (= 4,122 distinct emails) | an email pointing to **>1 different account** can’t identify a single company → a guessed company = a guessed manager | `0890612@shclc.com.cn` → accounts `001i000000Zv6FBAAZ` **and** `001i0000015woouAAA` |
| 4 | Deduplicate to one row per email | 2,782 | same email listed several times for the **same** account | — |

**Result: 134,202 clean bridge rows = 134,202 distinct, unambiguous emails.**

> Note on the arithmetic: DQ #1’s headline "4,122" counts *emails*; because each
> ambiguous email appears on multiple rows, it removes **8,884 rows**.
> `145,870 − 2 − 8,884 − 2,782 = 134,202`.

---

## 3. SFDC Accounts (`SFDC-Accounts-Data.csv`) — the account→SAP bridge

**Raw: 104,764 rows · 102 columns.**

### Columns: we keep 7 of 102
| Kept | Role |
|------|------|
| `Id` | join key from Contact `AccountId` |
| `SAP_Customer_Master_ID__c` | join key to SAP |
| `Name` | traceability (company name) |
| `Region__c`, `Industry`, `BillingCountry`, `Type` | enrichment / analysis (these turn out to be strong correlates of X1) |

**Dropped: 95 columns** — billing/shipping address detail, `SBQQ__*` (CPQ
billing config), `DOZISF__ZoomInfo_*`, geocode lat/long, audit fields, etc.
*(Reason: not join keys, not needed for Phase 1.)*

### Row cleaning & filtering
| # | Step | Rows affected | Reason |
|---|------|--------------:|--------|
| 1 | Normalize `account_id` (trim) | 0 changed | defensive — protect join against whitespace in future refreshes |
| 2 | Normalize `sap_id` (drop `.0`, strip leading zeros) | 0 changed *in this refresh* | defensive — earlier extracts had `0009201520` / `…​.0`; e.g. `0009201520 → 9201520` |
| 3 | Drop blank `account_id` | 0 | unusable |
| 4 | Deduplicate by `account_id` | 0 | one row per company |

**Result: 104,764 clean accounts**, of which **100,856 carry a SAP Customer ID**
and **3,908 do NOT** (those companies can’t reach SAP/X1 — they become a drop
reason for any user landing on them).

---

## 4. SAP (`SAP-Data.csv`) — the X1 source of truth

**Raw: 82,969 rows · 9 columns.** We keep 6 (`SAP Customer ID`, `X1`,
`X3 Name`, `X4 Name`, `X9 Name`, `0SALESORG_TD`); drop 3 (`Customer Name`,
`0SALESORG`, `SUBSEGMNT_TD` — redundant/empty).

### DQ #2 — exclude invalid / system X1 values
The `X1` column is free-text and sometimes holds a **status placeholder instead
of a person’s name**. We drop these (case-insensitive substring match on
`not assigned / unassigned / outdated / transferred`):

| X1 value (placeholder) | SAP rows removed |
|------------------------|-----------------:|
| Not assigned | 5,953 |
| TRANSFERRED TO EU CHANNEL PARTNER | 3,015 |
| UNASSIGNED ACCOUNT REP | 1,877 |
| TRANSFERRED TO US CHANNEL PARTNER | 1,568 |
| OUTDATED UNASSIGNED | 829 |
| UNASSIGNED SALES REP ASIA | 2 |
| UNASSIGNED CSR | 1 |
| **Total removed** | **13,245** |

**Result: 69,724 SAP customers with a valid, real X1 manager.**

---

## 5. Where the 4,854 external users end up (with examples)

This is the key table — **every external user, classified by outcome**:

| Outcome | Users | % | Reason | Example (email · company · country) |
|---------|------:|---:|--------|--------------------------------------|
| ✅ **LABELED** | **783** | 16.1% | full chain resolved to a valid X1 | `a.beer@biesterfeld.com` · Biesterfeld · Austria |
| ❌ No SFDC contact | 3,494 | 72.0% | email not present in Salesforce at all → no entry point | `1050746661@iran.ir` · Khorasan Powder Metallurgy · Iran |
| ❌ Account not in accounts file | 394 | 8.1% | contact found, but its `AccountId` isn’t in the accounts extract | `abazan@borgwarner.com` · Borgwarner · Spain |
| ❌ Ambiguous email (DQ #1) | 99 | 2.0% | email mapped to multiple accounts → removed for safety | `adam.bannerman@intralox.com` · Intralox · USA |
| ❌ X1 invalid/system (DQ #2) | 42 | 0.9% | company found in SAP but its X1 is `Not assigned`/`Transferred`/etc. | `anna@pbyplas.com` · PBY Plastics · USA |
| ❌ Account has no SAP id | 27 | 0.6% | account exists but `SAP_Customer_Master_ID__c` is blank | `anil.bhalla@gates.com` · Gates · Canada |
| ❌ SAP id not found in SAP | 15 | 0.3% | account’s SAP id doesn’t exist in the SAP file | `estelle.bourdon@nemera.net` · Nemera · France |
| | **4,854** | 100% | | |

### How to read this
- **The dominant loss (3,494 / 72%) is "no SFDC contact."** The user never
  entered Salesforce, so there is no chain to walk. *This is the single biggest
  lever:* improving the email match here would yield the most new labels.
- The next loss (394 / 8%) is an **extract-completeness** issue — the contact
  points to an account the accounts file doesn’t contain. Fixable with a fuller
  accounts export.
- The DQ-driven losses (99 ambiguous + 42 invalid X1 = 141, ~3%) are
  **deliberate quality choices** — we’d rather drop than mislabel, because the
  output feeds account-manager incentives.
- The SAP-side losses (27 + 15 = 42, <1%) are genuine **data gaps** in SAP.

---

## 6. Summary at a glance

The same outcomes, simplified to the three groups that matter:

| Outcome | % of users |
|---------|-----------|
| ✅ Got an X1 manager | 16% |
| ❌ No Salesforce contact at all | **72% ← the big one** |
| ❌ Other reasons (account missing, ambiguous email, no SAP id, invalid X1) | 12% |
| **Total** | **100%** |

And what that means for the model — the train / predict split:

| Group | Count | % of external users | What happens |
|-------|------:|--------------------:|--------------|
| ✅ Labeled (Phase 1 found their X1) | 783 | 16.1% | used to **train** the model |
| 🎯 Unlabeled (no X1 found) | 4,071 | 83.9% | this is what we **predict** on |
| **Total external users** | **4,854** | **100%** | |

---

## 7. Conclusion

The deterministic mapping (Chemille → SFDC Contact → SFDC Account → SAP), applied
with three data-quality controls, conclusively assigns an **X1 Account Manager to
783 of the 4,854 external Chemille users (16.1%)**. These 783 records are
**confirmed, fully traceable, and serve as the training data** for the Phase-2
prediction model.

The model’s task is therefore to predict X1 for the remaining **4,071 users
(≈84%)** that the joins could not resolve.

**A note on the denominator.** All percentages are measured against the **4,854
external users**, not the 6,301 raw rows. The **1,447 internal `@celanese.com`
employees are out of scope** for the entire exercise (they are not customers) and
are therefore neither trained on nor predicted.

**Framing of the approach.**
- **Train on:** 783 users — the 16% whose X1 we know with certainty.
- **Predict on:** 4,071 users — the 84% whose X1 is unknown.

This is the standard supervised-learning setup: a smaller, verified set teaches
the model to fill in the larger unknown set.

**Known constraint.** The 783 training examples are spread across **193 distinct
managers (~4 examples each)**, which is thin for predicting 4,071 users across a
potentially even larger set of managers. Phase 2 therefore relies primarily on
**`email_domain`** — the strongest available predictor (association 0.69 with X1)
— and may need to predict a **coarser target** (e.g. region or segment) and/or
**recover additional labels** by improving the email-match rate. The single
highest-impact improvement remains closing the **72% "no Salesforce contact"
gap**, which is a data-availability issue, not a limitation of the pipeline.

---

## 8. Phase-2 approaches — predicting X1 for the 4,071 unmapped users

We use the **783 labeled users as training data** to predict the X1 manager for
the **4,071 unlabeled users**. This is a supervised classification problem.

### 8.1 Inputs (features) — what we are allowed to use

The model may only use attributes that exist for **every** user, including the
unlabeled ones. The unlabeled users only have their **Chemille fields**, so the
valid inputs are:

| Input feature | Source | Association with X1 (Cramér's V) |
|---------------|--------|---------------------------------:|
| `email_domain` (e.g. `biesterfeld.com`) | derived from Chemille `Email` | **0.69 (strongest)** |
| `country` | Chemille | 0.53 |
| `company` (free text) | Chemille | 0.50 |

> **Not allowed as inputs (data leakage):** `account_region`, `account_industry`,
> `account_type`, `sap_salesorg`, SAP id, account id. These come from the SFDC/SAP
> join and only exist *after* a user is matched — they are unknown for the 4,071
> unlabeled users, so they cannot be used to predict them. (They are useful for
> *understanding* the data, e.g. region drives X1 at 0.86, but not as inputs.)

### 8.2 Outputs (what we can predict) — two options

| Option | Prediction target | Classes | Pros / Cons |
|--------|--------------------|--------:|-------------|
| **A. Direct** | the individual **X1 manager** | 193 | Exactly what the business wants, but hard: ~4 training examples per manager. |
| **B. Coarser (hierarchical)** | a **group** first — e.g. `region` / `sales segment` / company cluster — then resolve the manager within it | few | Far easier to learn (more examples per class); still useful; the individual manager can often be derived from the group + company. |

Recommendation: try **A** first as the goal, but fall back to **B** where A is
too uncertain (the 193-manager sparsity makes B a realistic safety net).

### 8.3 Modeling approaches (simple → advanced)

| # | Approach | How it works | Best for |
|---|----------|--------------|----------|
| 1 | **Domain lookup (rule-based)** | from the 783 labels build `email_domain → X1` (majority vote); apply to any unlabeled user sharing that domain | High-precision quick win. Justified by the 0.69 association — everyone at a domain usually shares one manager. |
| 2 | **Nearest-neighbour / similarity** | for an unlabeled user, find the most similar labeled user by `company` text + `domain` (TF-IDF or embeddings) and borrow their X1 | Handles the long tail (works even for managers with 1 example); the similarity score becomes a **confidence** value. |
| 3 | **Multiclass classifier** | Logistic Regression / Random Forest / Gradient Boosting on one-hot `domain` + `country` + TF-IDF `company` | A trainable baseline; weaker where classes have very few examples. |
| 4 | **Embedding / LLM semantic match** | embed company name + domain, match to the nearest labeled company cluster | Best for messy/duplicate company names (`Acme Inc` vs `ACME Incorporated`). |

### 8.4 Recommended pipeline (tiered, with human-in-the-loop)

1. **Tier 1 — domain lookup** (approach 1): assign high-confidence predictions
   where the domain is already known from the 783 labels.
2. **Tier 2 — similarity model** (approach 2): for the rest, predict from the
   closest labeled company/domain; the similarity score = confidence.
3. **Confidence threshold + human review:** auto-accept high-confidence
   predictions; route low-confidence ones to a person. Because the output feeds
   **account-manager incentives**, a wrong-but-confident prediction is costly —
   so thresholding matters more than raw accuracy.

### 8.5 How we will measure success

Hold out part of the **783 known users** as a test set, predict their X1, and
report **accuracy** and **per-class coverage**. This tells us how far we can trust
the model before applying it to the 4,071 unknowns.

### 8.6 Summary

- **Train on:** 783 labeled users (the 16% we know).
- **Predict on:** 4,071 unlabeled users (the 84% we don't).
- **Inputs:** `email_domain`, `country`, `company` (Chemille fields only).
- **Output:** the X1 manager (or a coarser region/segment, then resolve).
- **Constraint:** ~4 examples per manager → lean on `email_domain`, use
  confidence thresholds, and recover more labels by closing the 72% gap.

---

## 9. One-line summary for the deck

> From **4,854 external Chemille users**, deterministic joins (Chemille → SFDC
> Contact → SFDC Account → SAP) plus three data-quality controls produce
> **783 confirmed X1 labels (16.1%)**. The remaining 84% are unmapped —
> **72 points of which are simply users with no Salesforce contact record** —
> and become the prediction target for the Phase-2 ML model.
