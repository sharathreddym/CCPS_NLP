# Chemille User → X1 Account Manager — Analysis & Phase-1 Documentation

**Goal:** for every Chemille portal user (an external customer), determine their
**X1 Account Manager** so the mapping can drive account alignment and
account-manager incentives. Because no single system stores this link, we build
it in two phases:

- **Phase 1 (this work):** deterministic joins across Chemille → SFDC → SAP to
  produce a *known-correct* labeled dataset.
- **Phase 2 (next):** a machine-learning model that predicts X1 for the majority
  of users that joins cannot reach.

---

## 1. Data landscape

Three systems, four files. SFDC (Salesforce) is the only bridge between the
Chemille portal and SAP (where the X1 manager actually lives).

| File | Rows | Cols | Role |
|------|-----:|-----:|------|
| `Chemille-User-Data.csv` | 6,301 | 5 | Portal users (Email, firstName, lastName, country, company) |
| `SFDC-Contact-Data.csv` | 145,875 | 32 | **People** at customer companies (the email bridge) |
| `SFDC-Accounts-Data.csv` | 93,454 | 39 | **Companies** = the customers (hold the SAP id) |
| `SAP-Data.csv` | 82,969 | 9 | SAP customers + X1/X3/X4/X9 managers (**source of truth**) |

**CRM data model (important):** in Salesforce an **Account = the customer company**
and a **Contact = an individual person at that company**. The X1 manager is
assigned to the *company* (Account/SAP customer), not the person. So the chain
must go *person → their company → who manages that company*.

---

## 2. The mapping chain

```
CHEMILLE_USER.Email
      │  exact email match (normalized)
      ▼
SFDC_CONTACT.AccountId
      │  AccountId  ==  Account.Id  (case-sensitive 18-char)
      ▼
SFDC_ACCOUNT.SAP_Customer_Master_ID__c
      │  normalized SAP id
      ▼
SAP.[SAP Customer ID]
      ▼
X1 (+ X3 / X4 / X9)  Account Manager
```

All four join keys exist and are the correct columns.

---

## 3. Data Quality Controls (from the planning doc, page 2)

| # | Control | Implementation | Effect on this data |
|---|---------|----------------|---------------------|
| **1** | Remove emails linked to **multiple** SFDC accounts | group contacts by email; if it maps to >1 distinct `AccountId`, **drop all rows** for that email (never keep one — a guessed company = a guessed manager) | dropped **4,122** ambiguous emails |
| **2** | Exclude **invalid/system** assignments | `X1` is a free-text field; some rows hold placeholders instead of a name. Drop case-insensitive matches of `not assigned / unassigned / outdated / transferred` | dropped **13,245** SAP rows |
| **3** | **Normalize identifiers** across systems | apply identical cleaning to **both sides** of every join key | see §4 |

### DQ #2 — what the "invalid" values actually are
These are literal strings sitting **inside the `X1` column** (not a separate flag):

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

Substring matching (vs. an exact list) is deliberate — it catches long-tail
variants (`OUTDATED UNASSIGNED`, `UNASSIGNED CSR`) an exact list would miss.

---

## 4. Normalization map (DQ #3)

Normalize **only join keys**, and **identically on both sides**:

| Join key | Files (both sides) | Treatment |
|----------|--------------------|-----------|
| `email` | Chemille `Email` + Contact `Email` | trim → lowercase → strip leading `*` (SFDC bounce markers) |
| `account_id` | Contact `AccountId` + Account `Id` | **trim only — NOT lowercased** (SFDC 18-char Ids are case-sensitive) |
| `sap_id` | Account `SAP_Customer_Master_ID__c` + SAP `SAP Customer ID` | trim → drop trailing `.0` (float artifact) → strip leading zeros |

`X1` gets a `.strip()` but its **casing is preserved** (label reads `MARCO SECCHI`,
not `marco secchi`).

**Whitespace audit on `account_id`:** verified — `0` rows differ after `.strip()`
on either side; join overlap is `17,623` raw and `17,623` stripped (identical).
The `.strip()` is therefore defensive insurance against future data refreshes,
not a fix for today. (A handful of malformed-length id rows exist: 1 in contacts,
3 in accounts — they can't match a valid 18-char id and don't affect results.)

---

## 5. Phase-1 result — the funnel

Starting from external Chemille users (after dropping 1,447 internal `@celanese.com`):

| Stage | Users | % of external |
|-------|------:|--------------:|
| Chemille external users | 4,854 | 100.0% |
| → matched an SFDC contact (email) | 1,261 | 26.0% |
| → contact linked to an Account | 860 | 17.7% |
| → account had a SAP Customer ID | 833 | 17.2% |
| → **resolved to a VALID X1 (LABEL)** | **779** | **16.0%** |

- **779 labeled users** across **192 distinct X1 managers** → the Phase-1 training set.
- **4,075 users (84%)** remain unlabeled → these are the Phase-2 (ML) targets.

---

## 6. Key observations & conclusions

1. **The bottleneck is the email match (hop 1), not the SAP side.** Only **26%**
   of users match an SFDC contact — that single step caps Phase-1 coverage.
   Hops 3→4 (Account→SAP→X1) are healthy: 89,540 of 93,452 accounts carry a SAP
   id, and 55,502 account-side SAP ids exist in SAP. **To raise coverage, improve
   the email match — not the SAP join.**

2. **The data extract quality matters enormously.** An earlier extract had a
   near-broken contact→account link (only **121** shared ids) and produced **0**
   end-to-end labels. The current fuller extract shares **17,623** ids and the
   chain works. *Always sanity-check key overlap before trusting a run.*

3. **A direct Chemille-email → SAP shortcut is not viable** (per the doc and the
   data): many users aren't in SAP, company names are duplicated/inconsistent,
   and many-to-many links explode. The SFDC bridge is necessary.

4. **The label set is sparse and long-tailed.** 779 labels over 192 managers is
   **~4 users per manager** on average, and many managers have only 1. This is a
   real challenge for multi-class modeling — Phase 2 will likely need to (a)
   recover more labels by improving the email match, and/or (b) predict a coarser
   target (region/segment, or manager-via-company-cluster) rather than the exact
   individual.

5. **`X1` is dirty free-text**, mixing real names with system placeholders
   (13,245 rows). Treating those as labels would teach the model to "predict"
   `Not assigned` — DQ #2 is essential, not cosmetic.

6. **Governance is satisfied:** SAP = source of truth, SFDC = label generation
   only, the model = prediction only; every label is traceable
   (`chemille_email → sfdc_account_id → sap_customer_id → X1`), and only
   essential fields are carried forward.

---

## 7. Deliverables in this folder

| File | What it is |
|------|------------|
| `phase1_build_training_dataset.py` / `.ipynb` | Commented Phase-1 pipeline; outputs the labeled dataset |
| `chemille_user_x1_training_dataset.csv` | **Output:** 779 labeled rows (features → X1) |
| `eda_chemille_x1.py` / `.ipynb` | Exploratory analysis: funnel, distributions, class imbalance, **reusable Cramér's V association heatmap** |
| `ANALYSIS.md` | This document |

### Output dataset columns
`chemille_email, firstName, lastName, country, company, email_domain` (features) →
`X1_Account_Manager` (label) + `X3_Name, X4_Name, X9_Name` + traceability
(`sfdc_account_id, sfdc_account_name, sap_customer_id`).

---

## 8. Reusing the EDA for new columns

The EDA notebook's **Cramér's V heatmap** measures association between *any* two
categorical columns (0 = independent, 1 = perfectly associated). When you add a
new column to a source file:

1. Carry it through the merges (section 5 of the EDA notebook) so it lands on `df`.
2. Add it to `pick_categoricals(...)` (numeric columns → a Pearson `df.corr()` heatmap instead).
3. Re-run — the heatmap immediately shows how strongly the new column relates to
   **X1** and the other features, i.e. whether it's a useful Phase-2 predictor.

---

## 9. Recommended next steps

1. **Lift the 26% email match** — investigate non-matches (personal vs corporate
   domains, typos, missing contacts). Every recovered match is a free, exact label.
2. **Reconsider the prediction target** given ~4 labels/manager (coarser class or
   company-level prediction).
3. **Build Phase 2** on `chemille_user_x1_training_dataset.csv` with
   confidence thresholds + human-in-the-loop validation, as the doc requires.
