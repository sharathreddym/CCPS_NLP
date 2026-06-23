# Final Prediction Approach — Detailed, With Many Examples

This document explains, end to end and with **lots of real examples**, exactly how
the final model assigns an **X1 Account Manager** to a Chemille user. For every
step it states the **file(s) used**, the **column(s) used**, the **input**, and the
**output**. Nothing is hidden "under the hood".

---

## 0. Naming & file legend (used throughout)

| Term | Meaning | File on disk |
|---|---|---|
| **Chemille User file** | the portal users (external customers) | `Chemille-User-Data.csv` |
| **SFDC Contacts file** | individual **people** in Salesforce | `SFDC-Contact-Data.csv` |
| **SFDC Accounts file** | **companies** in Salesforce | `SFDC-Accounts-Data.csv` |
| **SAP Data file** | SAP customers + the **X1** manager | `SAP-Data.csv` |
| **email domain** | text **after the `@`** (e.g. `splast.pl`) — not a website | derived from an email |
| **X1** | the account-manager name we assign | column `X1` in the SAP Data file |

Rules:
- "contact" always means a row in the **SFDC Contacts file** (never a Chemille user).
- "domain" always means **email domain**.

---

## 1. The big picture

```
Chemille user
   │
   ├── PHASE 1: exact email match → confirmed X1            (783 users)
   │
   └── PHASE 2: the model — a 4-tier WATERFALL on the rest  (4,071 users)
          Tier 1  Enrichment            (email domain)
          Tier 2  Domain lookup         (email domain)
          Tier 3  Account-name match    (company name)
          Tier 4  Text similarity       (company+domain+country text)  ← fallback
```

The model tries **Tier 1 first**; if it isn't confident, it falls to Tier 2, then
3, then 4. **The first confident tier wins** and the later tiers are skipped.
Tier 4 always produces an answer, so every user ends with a prediction.

---

## 2. Shared data prep (done once, before any tier)

**Goal:** split the Chemille users into "already known" (783) and "to predict"
(4,071), and attach the true X1 to the known ones.

**Files & columns used:**
- `Chemille-User-Data.csv`: `Email`, `firstName`, `lastName`, `country`, `company`
- `SFDC-Contact-Data.csv`: `Email`, `AccountId`
- `SFDC-Accounts-Data.csv`: `Id`, `SAP_Customer_Master_ID__c`, `Name`
- `SAP-Data.csv`: `SAP Customer ID`, `X1`

**Steps & example:**
1. Normalize the Chemille `Email` (trim, lowercase, strip a leading `*`), take the
   **email domain**, and drop internal `@celanese.com` users.
   ```
   Email "AczubAT@splast.pl"  →  email = aczubat@splast.pl  →  email_domain = splast.pl
   ```
2. Run the Phase-1 chain (`Chemille email → SFDC Contact → SFDC Account → SAP → X1`).
   - If it resolves → user is **LABELED** (one of the 783).
   - If not → user is **UNLABELED** (one of the 4,071).

| | Users | meaning |
|---|---:|---|
| LABELED (`x1_true` known) | 783 | used to BUILD Tiers 2 & 4, and to VALIDATE all tiers |
| UNLABELED | 4,071 | the users the model predicts |

**Input:** the four raw files. **Output:** a table of 4,854 external users, each
with `email_domain`, `company`, `country`, and (for 783) `x1_true`.

---

## 3. TIER 1 — Enrichment (match key = email domain)

**Idea:** the user's exact email isn't in Salesforce, but a **colleague at the same
company** (same email domain) is. Look up that company's **real** manager.

**Files used:** `SFDC-Contact-Data.csv` + `SFDC-Accounts-Data.csv` + `SAP-Data.csv`
(to build the directory) and `Chemille-User-Data.csv` (the user being predicted).
**Columns used:** Contacts `Email`,`AccountId`; Accounts `Id`,`SAP_Customer_Master_ID__c`,`Name`; SAP `SAP Customer ID`,`X1`; Chemille `Email`.

### 3a. Build the directory  (input → output)
**Input:** every contact in the SFDC Contacts file.
For each **email domain**, find the **dominant account** its colleagues point to,
compute **purity** (share on that account), keep it only if purity ≥ 0.60 and the
account has a valid X1.
**Output:** a lookup table `email_domain → X1` (**6,847 clean domains**).

Real directory entries:
```
email_domain   #contacts  purity  dominant account (SFDC Accounts.Name)  →  X1 (SAP)
splast.pl          4        100%   SPLAST SP Z O.O.                       →  ADAM BARTOSZEK
verlan.es         10        100%   VERLAN S.A.                            →  JORDI ROCHER
utxgroup.com       3        100%   DURAFLEX                               →  YEW MENG (VINCENT) NG
```
*(Note `utxgroup.com`'s colleagues all point to the account "DURAFLEX" — the company
trades under a different legal name; the domain still resolves cleanly.)*

### 3b. Predict a user  (input → output)
**Input:** the user's email domain. **Output:** the directory's X1, or "skip".
- domain in directory → return its X1, confidence = **purity**.
- domain not in directory → **skip to Tier 2**.

Real Tier-1 assignments (from the output file):
```
chemille_email          email_domain   → X1                     confidence
aczubat@splast.pl       splast.pl      → ADAM BARTOSZEK         1.00
admin@verlan.es         verlan.es      → JORDI ROCHER          1.00
adm03.np@utxgroup.com   utxgroup.com   → YEW MENG (VINCENT) NG 1.00
akadyla@pl.nifco.com    pl.nifco.com   → GRZEGORZ MADAJCZYK    1.00
```
**Coverage: 727 users · held-out accuracy 97.7%.** This is the strongest tier, so
it runs first.

---

## 4. TIER 2 — Domain lookup (match key = email domain)

**Idea:** if *other Chemille users we already labeled* share this email domain, copy
their majority manager.

**Files used:** `Chemille-User-Data.csv` only (plus the `x1_true` labels from Phase 1).
**Columns used:** Chemille `Email` (for the domain) + the `x1_true` label.

### 4a. Build the lookup  (input → output)
**Input:** the 783 labeled users. Group them by **email domain**; take the
**majority** X1 and its **vote share**.
**Output:** a small table `email_domain → majority X1` (**~400 domains**).

Real lookup entries:
```
email_domain   #labeled users   majority X1        vote share
parker.com           1          TIMOTHY BUTURLA    100%
bd.com               1          NIELS OLSEN        100%
swoboda.com          1          ANDRAS ARVAI       100%
```

### 4b. Predict a user  (input → output)
**Input:** the user's email domain (only reached if Tier 1 skipped).
**Output:** majority X1 if the domain is known **and** vote share ≥ 0.80; else skip.

Real Tier-2 assignments:
```
chemille_email               email_domain   → X1                confidence(vote share)
9001093@parker.com           parker.com     → TIMOTHY BUTURLA   1.00
adhoward@parker.com          parker.com     → TIMOTHY BUTURLA   1.00
adrian.sanchez@swoboda.com   swoboda.com    → ANDRAS ARVAI      1.00
adriano.marangoni@bd.com     bd.com         → NIELS OLSEN       1.00
```
**Coverage: 440 users · accuracy 81.8%.**

> Why Tier 1 before Tier 2 (both use the domain): Tier 1's directory is built from
> ~145,000 contacts (6,847 domains, 97.7%); Tier 2's is built from only 783 users
> (~400 domains, 81.8%). Tier 1 is more accurate and wider, so it goes first; Tier 2
> only handles the few domains Tier 1's purity filter rejected.

---

## 5. TIER 3 — Account-name similarity (match key = company name)

**Idea:** the domain led nowhere (Tiers 1 & 2 failed), so switch signal: fuzzy-match
the user's **company name** to the full pool of SFDC account names, then resolve the
manager with a **purity vote**.

**Files used:** `SFDC-Accounts-Data.csv` + `SAP-Data.csv` (reference) and
`Chemille-User-Data.csv` (`company`). **Columns used:** Accounts `Name`,
`SAP_Customer_Master_ID__c`; SAP `SAP Customer ID`,`X1`; Chemille `company`.

### 5a. Build the reference  (input → output)
**Input:** every SFDC account with a valid X1. Group accounts by **normalized
company name**; for each name compute the **dominant manager** + **purity**.
**Output:** a reference of **41,461 company names**, each with `dominant_X1` and
`purity`. The names are TF-IDF-vectorized (character n-grams) so we can fuzzy-match.

Real reference entries (clean — purity 100%):
```
company name (SFDC Accounts.Name)   #accounts  purity   →  X1
VOGEL KUNSTSTOFFTECHNIK GMBH            1        100%    →  BEATRIX UFTRING
POLYHOSE KOHYEI INDIA PVT LTD           3        100%    →  PORURI, GOVINDANARAYANA
ILPEA                                   1        100%    →  RHONDA AGUILAR
```
Real **multi-manager** names (purity low → Tier 3 will NOT trust them):
```
PETER SOUS GMBH    purity 33%   {TIM LUEHRING:5, PABLO WILLMS:3, MANUEL HILGER:2, SANDRA FLOECK:2}
PROTO LABS INC     purity 13%   {RYAN BURNS:2, TYLER KITCHEN:2, CHRISTINA WEAR:2, TINA GLASS:1, ...}
```

### 5b. Predict a user  (input → output)
**Input:** the user's company name. **Output:** the matched company's dominant X1,
but only if **match cosine ≥ 0.85 AND purity ≥ 0.60**; else skip to Tier 4.
Two confidences guard it: *did we find the right company?* (cosine) and *does that
company map to one manager?* (purity).

Real Tier-3 assignments:
```
chemille_email                       company                       matched SFDC account name      → X1
a.kohl@vogel-kunststofftechnik.com   Vogel Kunststofftechnik GmbH  VOGEL KUNSTSTOFFTECHNIK GMBH   → BEATRIX UFTRING
acelano@ilpea.com                    ilpea                         ILPEA                          → RHONDA AGUILAR
ag@photogenic.co.uk                  Photo-gen-ic Ltd              PHOTO-GEN-IC LTD               → DESPINA THEODOROU
ahernandez@plastiexportsusa.com      Plastiexports                 PLASTIEXPORTS                  → ARTURO MELENDEZ-GOVEA
development@kohyei.in                Polyhose Kohyei               POLYHOSE KOHYEI INDIA PVT LTD  → PORURI, GOVINDANARAYANA
```
> The `development@kohyei.in` row is the key win: its domain `kohyei.in` was unknown
> (Tiers 1–2 failed), but its **company name** "Polyhose Kohyei" matched the SFDC
> account "POLYHOSE KOHYEI INDIA PVT LTD" → the **right** manager. (A plain text
> match would have wrongly matched "Kohyei Trading **Japan**" — see Tier 4.)

**Coverage: 492 users · accuracy 80.6%.**

---

## 6. TIER 4 — Text similarity, the fallback (match key = full text)

**Idea:** nothing above fired (unknown domain, company name didn't match an SFDC
account confidently). As a last resort, find the labeled Chemille user whose
**company+domain+country text** is most similar, and copy their X1.

**Files used:** `Chemille-User-Data.csv` only (plus `x1_true` labels). **Columns
used:** Chemille `company`, `Email` (domain), `country` + the `x1_true` label.

### 6a. Build  (input → output)
**Input:** the 783 labeled users. Build a text per user and TF-IDF-vectorize it.
```
text = company + " " + email_domain_name + " " + country
e.g.  "biesterfeld interowa biesterfeld austria"
```
**Output:** 783 reference text vectors, each tied to its `x1_true`.

### 6b. Predict  (input → output)
**Input:** the user's text. **Output:** the nearest labeled user's X1; confidence =
cosine similarity. `needs_review = True` if cosine < 0.50.

Real Tier-4 assignments:
```
chemille_email                  company                  → X1                conf   needs_review
a.principato@itwautomotive.com  ITW                      → OLIVIER ROUSSEAU  1.00   False
a.soliman@biesterfeld.com       Biesterfeld Interowa     → MERAL SEN         1.00   False
cweber@te.com                   TE Connectivity Germany  → HENRIK SCHMIDT    1.00   False
ceo@brandnewco.xyz              Some Tiny Startup        → KATARZYNA BEDNAR. 0.26   True   (weak → review)
```
- A cosine of **1.00** means the text is *identical* to a labeled user (e.g. another
  "Biesterfeld Interowa Austria" user) → reliable.
- A cosine of **0.26** ("Some Tiny Startup", brand-new domain, no similar company)
  is weak → flagged for human review.

**Coverage: 2,412 users · accuracy 37.9%** (the genuinely hard cases — which is why
most are flagged for review).

---

## 7. The waterfall in action — fall-through traces

### Example A — caught at Tier 1
```
aczubat@splast.pl  (company SPLAST, Poland)
  Tier 1: domain splast.pl IS in the enrichment directory (purity 100%) → ✓ ADAM BARTOSZEK  [STOP]
```

### Example B — caught at Tier 2
```
9001093@parker.com  (company Parker Hannifin Japan, Japan)
  Tier 1: domain parker.com NOT in enrichment directory      → skip
  Tier 2: domain parker.com IS in the 783 (vote share 100%)  → ✓ TIMOTHY BUTURLA  [STOP]
```

### Example C — caught at Tier 3
```
development@kohyei.in  (company Polyhose Kohyei, India)
  Tier 1: domain kohyei.in NOT in enrichment directory       → skip
  Tier 2: domain kohyei.in NOT among the 783                 → skip
  Tier 3: company "Polyhose Kohyei" matches SFDC account
          "POLYHOSE KOHYEI INDIA PVT LTD" (cosine 0.89, purity 100%) → ✓ PORURI, GOVINDANARAYANA  [STOP]
```

### Example D — falls all the way to Tier 4 (a multi-manager company)
```
a.soliman@biesterfeld.com  (company Biesterfeld Interowa, Austria)
  Tier 1: domain biesterfeld.com is SPLIT across many accounts → purity < 0.6 → NOT in directory → skip
  Tier 2: domain biesterfeld.com IS among the 783, but managers are split (vote share 0.33 < 0.80) → skip
  Tier 3: company name "Biesterfeld …" maps to several managers (purity < 0.6) → skip
  Tier 4: nearest text "biesterfeld interowa … austria" = another labeled Biesterfeld-Austria user
          (cosine 1.00) → ✓ MERAL SEN  [assigned, not flagged]
```

### Example E — falls to Tier 4 with LOW confidence (review)
```
ceo@brandnewco.xyz  (company Some Tiny Startup, Spain)
  Tier 1: domain unknown                → skip
  Tier 2: domain unknown                → skip
  Tier 3: company name matches nothing well (cosine < 0.85) → skip
  Tier 4: nearest text is only cosine 0.26 → ✓ (best guess) but needs_review = TRUE
```

---

## 8. Reading the output file — `chemille_users_x1_FINAL_all.csv`

One row per Chemille user:

| Column | Meaning | Example |
|---|---|---|
| `chemille_email`,`company`,`country`,`email_domain` | the user | `aczubat@splast.pl`, SPLAST, Poland, splast.pl |
| `X1_Account_Manager` | the assigned manager | ADAM BARTOSZEK |
| `tier` | which step assigned it | `Tier 1` |
| `source` | the descriptive label | `Tier 1 - Enrichment, email domain (confirmed)` |
| `confidence` | 0–1 (meaning depends on tier) | 1.0 |
| `needs_review` | `True` → check before using | False |

`confidence` meaning by tier: **Tier 1** = domain purity · **Tier 2** = vote share ·
**Tier 3** = cosine × purity · **Tier 4** = text cosine.

---

## 9. Using the model for a NEW user

`final_model_pipeline.ipynb` (or `.py`) builds all four references from the raw
files, then:
```python
predict_final_one(company="Parker Hannifin", email="newbuyer@parker.com", country="Japan")
# INPUT  : email_domain=parker.com, company="Parker Hannifin", country="Japan"
# Tier 1 : parker.com not in enrichment directory   → skip
# Tier 2 : parker.com in the 783 (vote 100%)         → TIMOTHY BUTURLA
# OUTPUT : TIMOTHY BUTURLA | method=2_domain_lookup_783 | confidence=1.0 | needs_review=False
```
The four source files must sit alongside the notebook (it reads them to build the
references).

---

## 10. Summary table

| Tier | Match key | Files used (reference) | Columns used | Confident when | Users | Acc. |
|---|---|---|---|---|---:|---:|
| 1 Enrichment | email domain | Contacts + Accounts + SAP | Email, AccountId, Id, SAP_Customer_Master_ID__c, Name, SAP Customer ID, X1 | domain in directory (purity ≥ .6) | 727 | 97.7% |
| 2 Domain lookup | email domain | the 783 labels (Chemille) | Email, x1 | vote share ≥ .8 | 440 | 81.8% |
| 3 Account-name | company name | Accounts + SAP | Name, SAP_Customer_Master_ID__c, SAP Customer ID, X1, company | match ≥ .85 & purity ≥ .6 | 492 | 80.6% |
| 4 Text (fallback) | company+domain+country text | the 783 labels (Chemille) | company, Email, country, x1 | always | 2,412 | 37.9% |

> **In one paragraph:** every Chemille user is pushed through a 4-tier waterfall.
> Tier 1 looks up the real manager via a colleague's email domain (best, 97.7%);
> Tier 2 reuses the 783 labels' domain majority; Tier 3 switches to fuzzy-matching
> the company name against SFDC accounts with a purity check; Tier 4 is a
> last-resort text match. The first confident tier wins, so each user gets the best
> available answer, with a confidence and a `needs_review` flag for safe use.
