# Data Transformation — Step by Step (Phase 1: 783 labels, then Enrichment: +727)

This document explains, with **real examples from the data**, exactly how we go
from raw files to:
- **Phase 1** → 783 confirmed X1 labels, and
- **Enrichment (Phase 1b)** → 727 additional X1 labels.

## Terminology used in this document (to avoid confusion)

| Term | Means | File |
|------|-------|------|
| **Chemille User file** | the portal users (external customers) we want to map | `Chemille-User-Data.csv` |
| **SFDC Contacts file** | individual **people** in Salesforce (CRM contacts) | `SFDC-Contact-Data.csv` |
| **SFDC Accounts file** | **companies** in Salesforce (the customers) | `SFDC-Accounts-Data.csv` |
| **SAP Data file** | SAP customers + their X1 account manager | `SAP-Data.csv` |
| **email domain** | the part of an email **after the `@`** (e.g. `biesterfeld.com`). NOT a website. | derived from an email address |
| **X1** | the account manager we are trying to find | lives in the SAP Data file |

> Whenever this document says "contact" it means a row in the **SFDC Contacts
> file** (a Salesforce person) — never a Chemille user. Whenever it says
> "domain" it means **email domain** (text after `@`).

---

# PART A — PHASE 1: getting the 783 labels

**Idea:** match a Chemille user to a Salesforce **person** by their *exact email*,
then walk that person → their company → SAP → the X1 manager.

## The chain (4 files, 3 joins)

```
Chemille User file   .Email
        │  JOIN 1: exact email match
        ▼
SFDC Contacts file   .Email  →  gives .AccountId
        │  JOIN 2: AccountId = Id
        ▼
SFDC Accounts file   .Id     →  gives .SAP_Customer_Master_ID__c
        │  JOIN 3: SAP id match
        ▼
SAP Data file        .[SAP Customer ID]  →  gives .X1
```

---

## Step A1 — Clean the Chemille User file

**File:** `Chemille-User-Data.csv` (6,301 rows)

Operations:
1. Normalize the `Email`: trim spaces, lowercase, strip a leading `*`.
2. Remove rows with a blank email.
3. Compute the **email domain** (text after `@`).
4. **Remove internal users** whose email domain is `celanese.com` (employees, not customers).
5. Keep one row per email (deduplicate).

Example of a removed internal user:
```
michael.hess@celanese.com   → email domain = celanese.com → REMOVED (internal)
```

**Result: 4,854 distinct external Chemille users.**

---

## Step A2 — Clean the SFDC Contacts file and apply DQ #1

**File:** `SFDC-Contact-Data.csv` (145,870 rows). We use only two columns:
`Email` and `AccountId`.

Operations:
1. Normalize the contact `Email` (same rule as above). Example fix:
   ```
   *david.price@selecteng.ca   →   david.price@selecteng.ca
   ```
2. Drop rows with blank `Email` or blank `AccountId`.
3. **DQ #1 — remove ambiguous emails.** If one email is linked to **more than
   one** `AccountId`, it cannot identify a single company, so we drop **all** of
   its rows. Real examples (each email points to 2 different Salesforce accounts):
   ```
   mariana.brito@arlington-automotive.com  → AccountIds 0016S00003ME3DjQAL  AND  001i000000Zv5ogAAB   → dropped
   sam.craig@convatec.com                  → AccountIds 0016S00002xhwz3QAA  AND  001i000000swmn2AAA   → dropped
   klawson@fitfibers.com                   → AccountIds 001i000000Zv4yeAAB  AND  001i000000swlvAAAQ   → dropped
   ```
4. Keep one row per email.

**Result: a clean lookup of `email → AccountId` (134,202 rows).**

---

## Step A3 — Clean the SFDC Accounts file

**File:** `SFDC-Accounts-Data.csv` (104,764 rows). Key columns: `Id` and
`SAP_Customer_Master_ID__c`.

Operations:
1. Normalize `Id` (trim only — these 18-char IDs are case-sensitive).
2. Normalize `SAP_Customer_Master_ID__c`: trim, drop a trailing `.0`, strip
   leading zeros (so it matches the SAP side). Illustrative:
   ```
   0009201520   →   9201520
   ```
3. Keep one row per `Id`.

**Result: a clean lookup of `AccountId → SAP Customer ID`.**

---

## Step A4 — Clean the SAP Data file and apply DQ #2

**File:** `SAP-Data.csv` (82,969 rows). Key columns: `SAP Customer ID` and `X1`.

Operations:
1. Normalize `SAP Customer ID` (same rule as the SAP id above).
2. **DQ #2 — remove invalid/system X1 values.** The `X1` column is free text and
   sometimes holds a status instead of a person's name. We drop these (real
   counts from the data):
   ```
   Not assigned                       → 5,953 rows   REMOVED
   TRANSFERRED TO EU CHANNEL PARTNER  → 3,015 rows   REMOVED
   UNASSIGNED ACCOUNT REP             → 1,877 rows   REMOVED
   TRANSFERRED TO US CHANNEL PARTNER  → 1,568 rows   REMOVED
   ```

**Result: a clean lookup of `SAP Customer ID → X1` (69,724 valid managers).**

---

## Step A5 — Join the chain and read off X1

Join the four cleaned tables in order. Here are **4 real Chemille users** traced
all the way through to their X1 manager:

| Chemille email (from Chemille User file) | Company | SFDC AccountId | SAP Customer ID | **X1 manager** |
|---|---|---|---|---|
| `a.beer@biesterfeld.com` | Biesterfeld | `0016S00003ME3CnQAL` | `1805083` | **MERAL SEN** |
| `a.dourado@biesterfeld.com` | Biesterfeld | `0016S00003D2bncQAB` | `1801011` | **JOSE GONZALEZ** |
| `a.fiorini@eltekgroup.it` | ELTEK spa | `001i000000Zv4gAAAR` | `1052267` | **EDMUNDO CAVALIERE** |
| `a.kulinska@biesterfeld.com.pl` | Biesterfeld Polska | `0016S00003ME1NTQA1` | `1802480` | **ANNA BURSAKOVA** |

> Note how the two `biesterfeld.com` users map to **different** AccountIds and
> therefore **different** managers — Biesterfeld is a large distributor with
> several account managers. This becomes important in Part B.

**Result of Part A: 783 Chemille users now have a confirmed X1 manager.**
These are the **Phase-1 labels** (also called the "training data").
The remaining **4,071** external users did NOT match a Salesforce contact by
exact email, so they have no X1 yet.

---

# PART B — ENRICHMENT (Phase 1b): getting 727 more labels

**The problem with the 4,071:** their *exact email* is not in the SFDC Contacts
file, so Part A's Join 1 fails for them.

**The new idea:** an exact person may be missing, but a **colleague at the same
company** is often present in the SFDC Contacts file. Colleagues share the same
**email domain** (e.g. everyone at `@ergotech.it`), and a company's X1 manager is
the same for the whole company. So we match on **email domain** (company-level)
instead of exact email (person-level).

## Step B1 — Build an "email-domain → X1" directory

We build a reference table using **three files only** here:
`SFDC-Contact-Data.csv` + `SFDC-Accounts-Data.csv` + `SAP-Data.csv`.
(The Chemille User file is NOT used to build the directory — it is used later, in
Step B2, as the thing we look up.)

For **every** contact in the SFDC Contacts file:
1. Take the contact's **email domain** (text after `@`).
2. Follow that contact's `AccountId` → SFDC Accounts → `SAP Customer ID` → SAP →
   `X1` (the same chain as Part A).
3. Group by email domain and pick the **dominant account** for that domain.

We add two safety filters:
- **Drop generic email domains** (`gmail.com`, `qq.com`, `hotmail.com`, …) —
  these are personal mailboxes, not companies.
- **Purity ≥ 60%** — the email domain must point *mostly* to one account.
  `purity = (contacts on the domain that point to the dominant account) ÷ (all
  contacts on that domain)`.

Real directory entries built this way:

| email domain | dominant Salesforce account (Name) | purity | **X1 manager** |
|---|---|---|---|
| `ergotech.it` | ERGOTECH S.R.L. | 93% | MICAELA DI GREGOLI |
| `snetor.com` | SNETOR FRANCE | 69% | PRANNOY VINCENT ALVA |
| `us.ennovi.com` | ENNOVI ADVANCED MOBILITY | 83% | BRYAN KESSLER |
| `itwautochina.com` | SHANGHAI ITW PLASTIC & M | 67% | YANCHAO (ALLAN) JIANG |
| `jppolymers.in` | J.P. POLYMERS PVT LTD | 60% | PORURI, GOVINDANARAYANA |

*(Example of what purity rejects: `gmail.com` would point to thousands of
different accounts — purity near 0% — so it is excluded.)*

## Step B2 — Apply the directory to the 4,071 unlabeled Chemille users

**File looked up:** `Chemille-User-Data.csv` (only the 4,071 still-unlabeled).
For each, take their **email domain** and look it up in the directory from B1.

Real examples — unlabeled Chemille users that received a borrowed X1 (these are
part of the 727):

| Chemille email (unlabeled) | email domain | **borrowed X1 manager** |
|---|---|---|
| `a.olmi@ergotech.it` | `ergotech.it` | MICAELA DI GREGOLI |
| `a.yasser@snetor.com` | `snetor.com` | PRANNOY VINCENT ALVA |
| `aaron.blancas@us.ennovi.com` | `us.ennovi.com` | BRYAN KESSLER |
| `aaronzhang@itwautochina.com` | `itwautochina.com` | YANCHAO (ALLAN) JIANG |
| `aayush.jain@jppolymers.in` | `jppolymers.in` | PORURI, GOVINDANARAYANA |

**Result of Part B: 727 additional Chemille users get an X1 manager.**

## Step B3 — Validate the method using the 783 known labels

We cannot directly check the 727 (we have no "true answer" for them). So we test
the **same directory** on the 783 Phase-1 users, whose true X1 we already know:

```
For each of the 783 known users:
   ignore their known answer
   look up their email domain in the directory
   compare the borrowed X1 to their TRUE X1
Result: 96.4% match  (on the 385 of them that had a directory entry)
```

Because the 727 go through the identical directory, we expect ~96% reliability —
so these are treated as **recovered labels**, not ML guesses.

---

# Summary

| | Phase 1 (Part A) | Enrichment (Part B) |
|---|---|---|
| Match key | **exact email** (person-level) | **email domain** (company-level) |
| Files used to find the manager | SFDC Contacts + SFDC Accounts + SAP Data | SFDC Contacts + SFDC Accounts + SAP Data |
| File supplying the users to label | Chemille User file | Chemille User file (the 4,071 left over) |
| New labels | **783** | **+727** |
| Quality | exact / deterministic | 96.4% validated |
| Combined | | **≈ 1,510 confirmed X1 labels** |

**One-line summary:** Phase 1 matches each Chemille user to a Salesforce person
by exact email and follows the chain to SAP for the X1 manager (783 labels). The
enrichment then rescues users whose exact email is missing by matching on their
**email domain** to a colleague's company in Salesforce (+727 labels, 96%
validated), nearly doubling the confirmed mappings.
