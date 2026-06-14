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

## 6. One-line summary for the deck

> From **4,854 external Chemille users**, deterministic joins (Chemille → SFDC
> Contact → SFDC Account → SAP) plus three data-quality controls produce
> **783 confirmed X1 labels (16.1%)**. The remaining 84% are unmapped —
> **72 points of which are simply users with no Salesforce contact record** —
> and become the prediction target for the Phase-2 ML model.
