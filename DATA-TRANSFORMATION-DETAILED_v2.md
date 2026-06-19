# Data Transformation — FULLY DETAILED, Step by Step

Every micro-step, the **exact file**, the **exact column(s)**, the **operation**,
and a **real before→after example** from the data. Covers:
- **Phase 1** → 783 confirmed X1 labels
- **Enrichment (Phase 1b)** → 727 additional X1 labels

---

## Naming rules used everywhere in this document

| Word used | Exact meaning | File on disk |
|-----------|---------------|--------------|
| **Chemille User file** | external portal customers we want to map | `Chemille-User-Data.csv` |
| **SFDC Contacts file** | individual **people** in Salesforce | `SFDC-Contact-Data.csv` |
| **SFDC Accounts file** | **companies** in Salesforce | `SFDC-Accounts-Data.csv` |
| **SAP Data file** | SAP customers + the **X1** manager | `SAP-Data.csv` |
| **email domain** | text **after the `@`** in an email (e.g. `ergotech.it`). NOT a website. | computed |
| **X1** | the account-manager name we want | column `X1` in the SAP Data file |

- "**contact**" always = a row in the **SFDC Contacts file** (never a Chemille user).
- "**domain**" always = **email domain** (text after `@`).

### The exact columns each file starts with (the ones we use)

| File | Columns we read |
|------|-----------------|
| `Chemille-User-Data.csv` | `Email`, `firstName`, `lastName`, `country`, `company` |
| `SFDC-Contact-Data.csv` | `Email`, `AccountId` (2 of its 119 columns) |
| `SFDC-Accounts-Data.csv` | `Id`, `SAP_Customer_Master_ID__c`, `Name` (+region/industry for analysis) |
| `SAP-Data.csv` | `SAP Customer ID`, `X1` (+X3/X4/X9) |

---

# PART A — PHASE 1 (783 labels): match by EXACT email

High-level chain (4 files, 3 joins):

```
Chemille User file.Email  ──exact email──►  SFDC Contacts file.Email
SFDC Contacts file.AccountId  ──equals──►  SFDC Accounts file.Id
SFDC Accounts file.SAP_Customer_Master_ID__c  ──equals──►  SAP Data file.[SAP Customer ID]
SAP Data file.X1  ──►  the answer
```

---

## STEP A1 — Clean the **Chemille User file**

**File:** `Chemille-User-Data.csv` · **Start: 6,301 rows.**

Raw sample (exact columns as they appear):
```
Email                         firstName  lastName        country        company
10356602@mackenzista.com.br   Bruna      Uemura Rucker   Brazil         Universidade Presbiteriana Mackenzie
1050746661@iran.ir            Ali        Bolouki         Iran           Khorasan Powder Metallurgy
115295@global.ul.com          Jacob      Farny           United States  Bumbleroot Farms Inc
```

**A1.1 — Normalize column `Email`.** Operation: trim spaces → lowercase → strip a
leading `*`. (In this file no emails had uppercase, so most are unchanged; the
rule still runs to be safe.) Creates a working column `email`.

**A1.2 — Drop rows where `email` is blank.** (No possible match without an email.)

**A1.3 — Compute `email_domain`** = the part of `email` after `@`.
```
email = 1050746661@iran.ir   →   email_domain = iran.ir
```

**A1.4 — Remove internal Celanese users** where `email_domain` == `celanese.com`.
Real removed example:
```
michael.hess@celanese.com   →   email_domain = celanese.com   →   REMOVED (employee, not a customer)
```
Removes **1,447** rows.

**A1.5 — Deduplicate** to one row per `email`.

> **Output of A1: 4,854 distinct external Chemille users.**

---

## STEP A2 — Clean the **SFDC Contacts file** + Data-Quality Control #1

**File:** `SFDC-Contact-Data.csv` · **Start: 145,870 rows.** Columns used:
`Email`, `AccountId`.

**A2.1 — Normalize column `Email`** (same rule). This strips Salesforce
"bounce-marker" `*` prefixes. Real before→after:
```
*david.price@selecteng.ca               →   david.price@selecteng.ca
*jeffrey.theisen@gcpaint.com            →   jeffrey.theisen@gcpaint.com
*kathleen.cerchio@quantum-polymers.com  →   kathleen.cerchio@quantum-polymers.com
```

**A2.2 — Normalize column `AccountId`** (trim spaces only; these 18-character IDs
are case-sensitive so we do NOT lowercase them).

**A2.3 — Drop rows** where `Email` or `AccountId` is blank. (Removes 2 rows.)

**A2.4 — DATA-QUALITY CONTROL #1: remove ambiguous emails.**
Operation: group by `email`, count the number of **distinct** `AccountId`s.
If an email maps to **more than one** account, it cannot identify a single
company, so we drop **every** row for that email. Real examples (each maps to 2
different Salesforce accounts):
```
mariana.brito@arlington-automotive.com  →  AccountId 0016S00003ME3DjQAL  AND  001i000000Zv5ogAAB  →  DROP all
sam.craig@convatec.com                  →  AccountId 0016S00002xhwz3QAA  AND  001i000000swmn2AAA  →  DROP all
klawson@fitfibers.com                   →  AccountId 001i000000Zv4yeAAB  AND  001i000000swlvAAAQ  →  DROP all
```
This removes **8,884 rows** (covering **4,122 distinct emails**).

**A2.5 — Deduplicate** to one row per `email`.

> **Output of A2: a clean lookup table `email → AccountId` (134,202 rows).**

---

## STEP A3 — Clean the **SFDC Accounts file**

**File:** `SFDC-Accounts-Data.csv` · **Start: 104,764 rows.** Columns used:
`Id`, `SAP_Customer_Master_ID__c`, `Name`.

Raw sample:
```
Id                  SAP_Customer_Master_ID__c   Name
001UH00000tHM7hYAG  9201520                     INTERNATIONAL REFRESHMENTS INDIA PVT LTD
001UH00000t1rSPYAY  9201510                     GFR MBH
001UH00000sslakYAA  9201506                     DISTRIBUIDORA CHEMSOL DE COSTA RICA
```

**A3.1 — Normalize column `Id`** (trim only; case-sensitive). This is the key
that matches `AccountId` from A2.

**A3.2 — Normalize column `SAP_Customer_Master_ID__c`.** Operation: trim → drop a
trailing `.0` → strip leading zeros, so it matches the SAP side's format.
Illustrative (the current file is already clean, so this is defensive):
```
0009201520   →   9201520
9201520.0    →   9201520
```
This becomes the working column `sap_id`.

**A3.3 — Deduplicate** to one row per `Id`.

> **Output of A3: a clean lookup table `AccountId → sap_id` (+ company `Name`).**

---

## STEP A4 — Clean the **SAP Data file** + Data-Quality Control #2

**File:** `SAP-Data.csv` · **Start: 82,969 rows.** Columns used:
`SAP Customer ID`, `X1`.

Raw sample (note row 2 is a status, not a person):
```
SAP Customer ID   X1
2806507           MARCO SECCHI
2808282           TRANSFERRED TO US CHANNEL PARTNER
2808435           RICARDO DELGADO
```

**A4.1 — Normalize column `SAP Customer ID`** (same trim / drop `.0` / strip
leading zeros) → working column `sap_id`. This matches A3's `sap_id`.

**A4.2 — Clean column `X1`** (trim spaces; keep the original capitalization of
the name).

**A4.3 — DATA-QUALITY CONTROL #2: remove invalid / system X1 values.**
Operation: lowercase a copy of `X1` and drop any row containing
`not assigned`, `unassigned`, `outdated`, or `transferred`. Real counts removed:
```
Not assigned                       → 5,953 rows  REMOVED
TRANSFERRED TO EU CHANNEL PARTNER  → 3,015 rows  REMOVED
UNASSIGNED ACCOUNT REP             → 1,877 rows  REMOVED
TRANSFERRED TO US CHANNEL PARTNER  → 1,568 rows  REMOVED
(… 13,245 rows removed in total)
```

**A4.4 — Deduplicate** to one row per `sap_id`.

> **Output of A4: a clean lookup table `sap_id → X1` (69,724 valid managers).**

---

## STEP A5 — Join the four cleaned tables (3 joins)

We now chain the lookups. Worked end-to-end for the Chemille user
`a.beer@biesterfeld.com`:

| Stage | File / lookup used | Key value | Value obtained |
|-------|--------------------|-----------|----------------|
| start | Chemille User file | `email = a.beer@biesterfeld.com` | — |
| JOIN 1 | A2 table `email → AccountId` | `a.beer@biesterfeld.com` | `AccountId = 0016S00003ME3CnQAL` |
| JOIN 2 | A3 table `AccountId → sap_id` | `0016S00003ME3CnQAL` | `sap_id = 1805083` |
| JOIN 3 | A4 table `sap_id → X1` | `1805083` | **`X1 = MERAL SEN`** |

Four more real users traced the same way:

| Chemille email | AccountId (SFDC) | sap_id (SAP) | **X1** |
|---|---|---|---|
| `a.beer@biesterfeld.com` | `0016S00003ME3CnQAL` | `1805083` | **MERAL SEN** |
| `a.dourado@biesterfeld.com` | `0016S00003D2bncQAB` | `1801011` | **JOSE GONZALEZ** |
| `a.fiorini@eltekgroup.it` | `001i000000Zv4gAAAR` | `1052267` | **EDMUNDO CAVALIERE** |
| `a.kulinska@biesterfeld.com.pl` | `0016S00003ME1NTQA1` | `1802480` | **ANNA BURSAKOVA** |

> Two `biesterfeld.com` users → **different** AccountIds → **different** managers.
> Biesterfeld is a big distributor with several managers. (Remember this for Part B.)

> **Output of PART A: 783 Chemille users now have a confirmed X1 = the Phase-1 labels.**
> The other **4,071** external users failed JOIN 1 (their exact email is not in
> the SFDC Contacts file) → no X1 yet.

---

# PART B — ENRICHMENT / Phase 1b (+727 labels): match by EMAIL DOMAIN

**Why Part A missed 4,071 users:** their *exact email* is not in the SFDC Contacts
file. **New idea:** a **colleague at the same company** is often in the SFDC
Contacts file, and colleagues share an **email domain** (e.g. everyone at
`@ergotech.it`). A company's X1 is the same for the whole company → so match on
**email domain** (company-level) instead of exact email (person-level).

The directory in B1 is built from **3 files**: `SFDC-Contact-Data.csv` +
`SFDC-Accounts-Data.csv` + `SAP-Data.csv`. The **Chemille User file is used only
in B2** (the users we look up).

---

## STEP B1 — Build the `email_domain → X1` directory

**B1.1 — Start from the *normalized* SFDC Contacts** (`email`, `AccountId` with
blanks removed). **Important:** we do **NOT** apply Step A2's **DQ #1
(ambiguous-email removal)** or its **per-email dedup** here. Those were tuned for
Phase 1's *exact-email* match (one email → one answer). The domain directory needs
**every colleague row** for each email domain so it can find the dominant account
and compute purity, so it deliberately uses this fuller, pre-DQ#1 set.

> **Does keeping ambiguous emails distort the directory? No.** An ambiguous email
> (one email → 2 accounts) just becomes a minority "vote" inside its email domain.
> The **purity gate** (B1.4) absorbs it: if such votes are a small minority, the
> dominant account still wins (purity stays high → domain kept); if they make a
> domain genuinely split, purity falls below 60% → the whole domain is rejected.
> Verified empirically: building the directory with vs. without DQ #1 gives nearly
> identical quality — 727 labels @ 96.4% (kept, what we ship) vs. 781 @ 95.8%
> (DQ#1 applied). So purity, not DQ #1, is the safeguard at the domain level.

**B1.2 — Add `email_domain`** = text after `@` of each contact's `email`.

**B1.3 — Group by `email_domain`** and, for each domain, compute:
- number of contacts on that domain,
- number of **distinct** `AccountId`s,
- the **dominant** `AccountId` (most frequent),
- **purity** = (contacts on the dominant account) ÷ (all contacts on the domain).

### Worked purity example — email domain `ergotech.it`
All SFDC Contacts at this email domain (column `email`, column `AccountId`):
```
email                          AccountId
a.serrajotto@ergotech.it       001i000000kUO0fAAG
claudia.molinatti@ergotech.it  001i000000kUO0fAAG
d.bariona@ergotech.it          001i000000kUO0fAAG
f.cobetto@ergotech.it          001i000000kUO0fAAG
f.fabbri@ergotech.it           001i000000kUO0fAAG   (x3)
g.gaida@ergotech.it            001i000000kUO0fAAG
l.perrucchione@ergotech.it     001i000000kUO0fAAG
m.colle@ergotech.it            001i000000kUO0fAAG
m.dini@ergotech.it             001i000000kUO0fAAG   (x3)
s.agosti@ergotech.it           001i000000kUO0fAAG
s.bacchini@ergotech.it         001i000000Zv5NwAAJ   ← the 1 odd one
```
- distinct accounts = **2**
- dominant = `001i000000kUO0fAAG` → appears **14 of 15** rows
- **purity = 14 / 15 = 93%**

**B1.4 — Keep only trustworthy domains:**
- **Drop generic email domains** (`gmail.com`, `qq.com`, `hotmail.com`, …) — personal mailboxes, not companies.
- **Keep purity ≥ 60%** — domain must point mostly to one account.

**B1.5 — Walk the dominant account to X1** (reusing A3 and A4 lookups):
```
dominant AccountId 001i000000kUO0fAAG
   → SFDC Accounts file: Name = ERGOTECH S.R.L., sap_id = 1075004
   → SAP Data file:      X1   = MICAELA DI GREGOLI
```
So the directory entry is:
```
ergotech.it  →  X1 = MICAELA DI GREGOLI   (purity 93%, account ERGOTECH S.R.L.)
```

Five real directory entries built this way:

| email_domain | dominant account `Name` | purity | sap_id | **X1** |
|---|---|---|---|---|
| `ergotech.it` | ERGOTECH S.R.L. | 93% | 1075004 | MICAELA DI GREGOLI |
| `snetor.com` | SNETOR FRANCE | 69% | — | PRANNOY VINCENT ALVA |
| `us.ennovi.com` | ENNOVI ADVANCED MOBILITY | 83% | — | BRYAN KESSLER |
| `itwautochina.com` | SHANGHAI ITW PLASTIC & M | 67% | — | YANCHAO (ALLAN) JIANG |
| `jppolymers.in` | J.P. POLYMERS PVT LTD | 60% | — | PORURI, GOVINDANARAYANA |

> **Output of B1: a big directory `email_domain → X1` (~30,878 clean domains).**

---

## STEP B2 — Apply the directory to the 4,071 unlabeled Chemille users

**File looked up:** `Chemille-User-Data.csv` (the 4,071 still without an X1).

**B2.1 — For each unlabeled user, take their `email_domain`** (from A1.3).
**B2.2 — Look it up in the B1 directory.** If found → borrow that X1.

Real unlabeled users that received a borrowed X1 (part of the 727):

| Chemille email (unlabeled) | email_domain | directory match | **borrowed X1** |
|---|---|---|---|
| `a.olmi@ergotech.it` | `ergotech.it` | ERGOTECH S.R.L. (93%) | MICAELA DI GREGOLI |
| `a.yasser@snetor.com` | `snetor.com` | SNETOR FRANCE (69%) | PRANNOY VINCENT ALVA |
| `aaron.blancas@us.ennovi.com` | `us.ennovi.com` | ENNOVI (83%) | BRYAN KESSLER |
| `aaronzhang@itwautochina.com` | `itwautochina.com` | SHANGHAI ITW (67%) | YANCHAO (ALLAN) JIANG |
| `aayush.jain@jppolymers.in` | `jppolymers.in` | J.P. POLYMERS (60%) | PORURI, GOVINDANARAYANA |

> Notice `a.olmi@ergotech.it` was NOT in the SFDC Contacts file as a person — but
> 15 of his colleagues were, so his company (`ergotech.it`) is in the directory
> and he inherits ERGOTECH's manager.

> **Output of B2: 727 additional Chemille users get an X1.**

---

## STEP B3 — Validate the method on the 783 known labels

We have no "true answer" for the 727, so we test the **directory (the
`email_domain → X1` lookup table from B1)** against the 783 Phase-1 users, whose
true X1 we already know. The directory is our *enrichment tool*; the 783 are our
*answer key*. We check: when the tool gives an answer, is it right?

```
For each of the 783 known users:
    look up their email_domain in the B1 directory
       → if the domain is a row in the directory, it returns a "borrowed" X1
    compare borrowed X1  vs  their TRUE X1
Result: 96.4% match  (on the 385 users the directory could answer)
```

### Why only 385 of the 783 (and not all 783)?

**Having an X1 and being in the directory are two independent things**, decided by
two different methods:

| | decided by | who passes |
|---|---|---|
| Has a true X1 | Phase 1 **exact-email** match (person-level) | all **783** |
| Is in the directory | the **email-domain purity** filter (company-level) | only **385** |

So a user can have a known X1 but live at an email domain that is *too messy* for
the domain tool to touch. Breakdown of the **398** that the directory could NOT
answer:

| Reason the email_domain is not in the directory | Count |
|---|---|
| Domain is **split across multiple accounts** (purity < 60%) | 388 |
| Dominant account has no valid X1 | 9 |
| Generic domain (gmail-type) | 1 |
| Domain not in SFDC Contacts at all | 1 |
| **Total not in directory** | **398** |

**Concrete example — `biesterfeld.com`.** In Step A5 we saw Biesterfeld users map
to *several* managers (`a.beer → MERAL SEN`, `a.dourado → JOSE GONZALEZ`,
`a.kulinska → ANNA BURSAKOVA`):
- Each user **has a true X1** (via exact email) → they are part of the **783**. ✅
- But the domain `biesterfeld.com` is **split across many accounts/managers**, so
  its **purity is below 60%** → the directory **deliberately excludes it** (it
  cannot honestly say "biesterfeld.com → one manager"). ❌
- So Biesterfeld labeled users are known-X1 users whose **domain is not in the
  directory** → they fall into the **398**, and are not used to score the tool.

### How to read the 96.4%

```
783 labeled users (all have a true X1)
   ├─ 385  domain IS in the directory  → tool answers → 96.4% match the truth  ← the score
   └─ 398  domain NOT in the directory → tool stays silent → cannot be scored
```

The directory is a **conservative tool**: it only answers for clean,
single-company domains, and we can only grade it where it actually answers. On
those 385 it is **96.4% correct**. Crucially, for the messy domains it stays
silent here — and it would stay equally silent for similar messy domains among the
4,071, so it never makes a risky guess. That silence is a safety feature, not a
failure.

Because the 727 recovered users pass through the **identical** directory under the
**same** purity filter, we expect ~96% reliability on them too → they are treated
as **recovered labels**, not ML guesses.

### Output files produced by this step

| File | Contents |
|------|----------|
| `domain_x1_directory.csv` | the B1 lookup table: `email_domain, X1, purity, n_contacts, n_accounts, dominant_account_id, dominant_account_name, sap_id` (6,847 rows) |
| `phase1b_recovered_727_labels.csv` | the 727 recovered users: `chemille_email, …, email_domain, recovered_X1, match_purity, dominant_account_name` |

*(Both are produced by the script `phase1b_domain_enrichment.py`.)*

---

# OVERALL SUMMARY

| | PART A — Phase 1 | PART B — Enrichment |
|---|---|---|
| Match key | **exact email** (person) | **email domain** (company) |
| Files to find the manager | SFDC Contacts → SFDC Accounts → SAP Data | SFDC Contacts → SFDC Accounts → SAP Data |
| File supplying users to label | Chemille User file (all 4,854) | Chemille User file (the 4,071 left over) |
| Quality control | DQ#1 (ambiguous email), DQ#2 (invalid X1) | generic-domain filter + purity ≥ 60% |
| New labels | **783** | **+727** |
| Validation | deterministic (exact) | **96.4%** vs the 783 |
| **Combined total** | | **≈ 1,510 confirmed X1 labels** |

**In one paragraph:** Phase 1 cleans the four files, matches each Chemille user to
a Salesforce **person** by *exact email* (`Chemille Email → SFDC Contact.Email`),
then follows `SFDC Contact.AccountId → SFDC Account.Id → SFDC Account.SAP id →
SAP.[SAP Customer ID] → SAP.X1`, removing ambiguous emails (DQ#1) and invalid
managers (DQ#2) — yielding **783** confirmed labels. The enrichment then rescues
users whose exact email is missing by matching on their **email domain** to a
colleague's company already in Salesforce (filtered to clean, single-company
domains), borrowing that company's manager — adding **727** labels at **96.4%**
validated accuracy, for **≈1,510** total.
