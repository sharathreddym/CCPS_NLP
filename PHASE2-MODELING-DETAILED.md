# Phase 2 — Predicting X1 (Detailed): how we used the 783 labels

This document explains, step by step and with real examples, what Phase 2 does:
**use the 783 labeled users (the Phase-1 output) to predict the X1 account
manager for the 4,071 unlabeled users.**

## Terminology (same as the other documents)
- **Chemille User file** = `Chemille-User-Data.csv` (the portal customers).
- **email domain** = text after the `@` (e.g. `parker.com`). Not a website.
- **X1** = the account-manager name we predict.
- **labeled users (783)** = users whose true X1 we found in Phase 1 → the "answer key".
- **unlabeled users (4,071)** = users with no X1 yet → what we predict.

## Code & output files this describes
| File | What it is |
|------|------------|
| `phase2_predict_x1.py` / `.ipynb` | Approaches 1 + 2 + tiered + evaluation |
| `phase2_embedding_match.py` / `.ipynb` | Approach 4 (embeddings), benchmarked vs 1+2 |
| `chemille_user_x1_phase2_predictions.csv` | 4,071 predictions (tiered model) |
| `chemille_user_x1_phase2_embedding_predictions.csv` | 4,071 predictions (embedding model) |

---

## 1. The setup: inputs and output

Phase 2 is a **supervised classification** problem.

**Inputs (features)** — only fields that exist for *every* user (including the
4,071), all taken from the **Chemille User file**:

| Feature | Example | Why it is allowed |
|---------|---------|-------------------|
| `email_domain` | `parker.com` | strongest signal (assoc. 0.69 with X1) |
| `country` | `Japan` | available for all users (0.53) |
| `company` | `Parker Hannifin Japan Ltd.` | available for all users (0.50) |

> We may NOT use account/SAP fields (region, sap_id, …) as inputs — they are blank
> for the 4,071 (that is "leakage"). See `ANALYSIS.md`.

**Output** — for each unlabeled user:

| Output column | Example |
|---------------|---------|
| `predicted_X1` | `TIMOTHY BUTURLA` |
| `confidence` | `1.0` |
| `method` | `domain_lookup` or `similarity` |
| `needs_review` | `True` if confidence < 0.50 |

---

## 2. How the data is prepared (same first step in every script)

1. Rebuild the merged table from the four raw files (the Phase-1 chain), giving
   each of the 4,854 external users their `email_domain`, `country`, `company`,
   and (for 783 of them) the true `x1`.
2. Split into **labeled** (783, have `x1`) and **unlabeled** (4,071, no `x1`).
3. Build a single text field used by the similarity methods:
   `text = company + " " + email_domain_name + " " + country`
   e.g. `"parker hannifin japan ltd. parker japan"`.

---

## 3. Approach 1 — Domain lookup (rule-based)

**Idea:** users at the same `email_domain` usually share the same manager.

**Fit (learn from the 783):** group the labeled users by `email_domain`; for each
domain take the **majority** X1 and its **vote share** (= confidence).
```
biesterfeld.com → most common X1 among labeled biesterfeld users  (confidence = its share)
parker.com      → TIMOTHY BUTURLA  (confidence = share of parker labeled users with him)
```

**Predict:** for an unlabeled user, look up their `email_domain`.
- domain known → return its majority X1 + vote share.
- domain unknown → **abstain** (no prediction; handed to Approach 2).

**Strength / weakness:** very precise where it fires, but only covers domains that
appeared among the 783 labels → limited coverage.

### 3.1 How "most common X1", "share", and "confidence" are calculated

In code (`fit_domain_lookup`) it is three lines:

```python
g     = train.groupby("email_domain")["x1"]
top   = g.agg(lambda s: s.value_counts().idxmax())            # most common X1
share = g.agg(lambda s: s.value_counts(normalize=True).max()) # vote share = confidence
```

For one domain:

- **most common X1** = the manager that appears most often among that domain's
  labeled users (`value_counts().idxmax()` — pandas counts each manager, then picks
  the one with the highest count).
- **share = confidence** = that top manager's count divided by the total number of
  labeled users at the domain:

```
confidence = (labeled users at the domain whose X1 = the most common X1)
             ----------------------------------------------------------
             (total labeled users at the domain)
```

So confidence is literally "what fraction of this domain's known users agree on the
same manager." 1.0 means perfect agreement; 0.5 means the domain is split.

#### Example A — clean domain (one manager) → confidence 1.0

Among the 783 labels, suppose 5 users have `@parker.com`:

| labeled user | true X1 |
|---|---|
| 9001093@parker.com | TIMOTHY BUTURLA |
| j.smith@parker.com | TIMOTHY BUTURLA |
| k.tanaka@parker.com | TIMOTHY BUTURLA |
| m.lee@parker.com | TIMOTHY BUTURLA |
| r.gomez@parker.com | TIMOTHY BUTURLA |

`value_counts()` → `{TIMOTHY BUTURLA: 5}`
- most common X1 = **TIMOTHY BUTURLA**
- share = 5 / 5 = **1.0** → confidence **1.0**

Result stored: `parker.com → (TIMOTHY BUTURLA, 1.0)`. Any unlabeled `@parker.com`
user gets TIMOTHY BUTURLA at confidence 1.0. Because 1.0 ≥ the 0.80 gate, the tiered
model accepts this via `domain_lookup` (matches the real row in §9).

#### Example B — split domain (several managers) → low confidence, gated out

Suppose 10 labeled users have `@biesterfeld.com`:

| X1 | count |
|---|---|
| ANNA BURSAKOVA | 4 |
| JOSE GONZALEZ | 3 |
| MARK FISCHER | 2 |
| LISA WONG | 1 |
| **total** | **10** |

`value_counts()` is ordered `[ANNA BURSAKOVA:4, JOSE GONZALEZ:3, …]`
- most common X1 = **ANNA BURSAKOVA** (highest count, 4)
- share = 4 / 10 = **0.40** → confidence **0.40**

Result stored: `biesterfeld.com → (ANNA BURSAKOVA, 0.40)`. The rule still *has* an
answer, but only 40% of Biesterfeld's known users actually map to it — unreliable.
Because 0.40 < 0.80, the **tiered model does NOT use domain lookup here**; it falls
through to similarity (Approach 2). That is exactly why the real Biesterfeld rows in
§9 show `method = similarity` and even disagree with each other.

#### Example C — moderate majority → fires alone, but gated out of the tiered model

Suppose 3 labeled users have `@acme.com`: MANAGER_A ×2, MANAGER_B ×1.
- most common X1 = **MANAGER_A**
- share = 2 / 3 = **0.667** → confidence **0.667**

Approach 1 on its own would return MANAGER_A at 0.667. But 0.667 < 0.80, so the
**tiered** model again prefers similarity. (0.80 was chosen precisely to reject
"2-out-of-3"-style weak majorities.)

#### Example D — domain seen only once → confidence 1.0 (a caveat)

If a domain appears for just **one** labeled user:
- most common X1 = that single user's manager
- share = 1 / 1 = **1.0** → confidence **1.0**

The math is correct, but the "1.0" rests on a single example, so it is *confident by
construction*, not by weight of evidence. Many small domains land here, which is part
of why high-confidence domain hits still aren't perfect on held-out data.

#### Quick reference

| Domain | counts | most common X1 | share = confidence | tiered uses it? (gate 0.80) |
|---|---|---|---:|---|
| parker.com | A:5 | A | 5/5 = **1.00** | ✅ domain_lookup |
| snetor.com | P:6 | P | 6/6 = **1.00** | ✅ domain_lookup |
| acme.com | A:2, B:1 | A | 2/3 = **0.67** | ❌ → similarity |
| biesterfeld.com | 4/3/2/1 | A | 4/10 = **0.40** | ❌ → similarity |
| onelabel.com | A:1 | A | 1/1 = **1.00** | ✅ (but 1 example only) |

> Tie note: if two managers tie for the top count, `value_counts()` keeps them in
> first-seen order and `idxmax()` returns the first — so ties are broken arbitrarily.
> Such domains have share ≤ 0.50 anyway, so the 0.80 gate discards them in the tiered
> model regardless.

### 3.2 The SAME calculation on the REAL 783 labeled users

The examples above used round "suppose" numbers. Here are the actual figures
computed from `chemille_user_x1_training_dataset.csv` (the 783 labels), grouped by
`email_domain`.

**Real CLEAN domains — every labeled user agrees → confidence 1.0**

| email_domain | # labeled users | X1 counts | most common X1 | share = confidence |
|---|---:|---|---|---:|
| `formerra.com` | 52 | AMINE OURABAH ×52 | AMINE OURABAH | 52/52 = **1.00** |
| `omya.com` | 13 | MARTINA MENZEL ×13 | MARTINA MENZEL | 13/13 = **1.00** |
| `resinex.com` | 11 | KATARZYNA BEDNARSKA ×11 | KATARZYNA BEDNARSKA | 11/11 = **1.00** |
| `entecresins.com` | 6 | GILBERT MUZQUIZ ×6 | GILBERT MUZQUIZ | 6/6 = **1.00** |

e.g. all 52 `@formerra.com` labeled users have the same manager, so
`value_counts() = {AMINE OURABAH: 52}` → most common = AMINE OURABAH, share =
52/52 = **1.0**. Any unlabeled `@formerra.com` user is assigned AMINE OURABAH at
confidence 1.0, and (1.0 ≥ 0.80) the tiered model accepts it via `domain_lookup`.

**Real SPLIT domains — labeled users disagree → low confidence → gated out**

| email_domain | # labeled users | X1 counts (real) | most common X1 | share = confidence |
|---|---:|---|---|---:|
| `kdfeddersen.com` | 32 | CHRIS SASSMANNSHAUSEN ×18, DESPINA THEODOROU ×14 | CHRIS SASSMANNSHAUSEN | 18/32 = **0.56** |
| `biesterfeld.com` | 30 | PRANNOY VINCENT ALVA ×10, ANNA BURSAKOVA ×9, MERAL SEN ×7, JOSE GONZALEZ ×4 | PRANNOY VINCENT ALVA | 10/30 = **0.33** |
| `inabata.com` | 16 | OHKE HAJIME ×5, QIANLI XIE ×5, … (7 managers) | OHKE, HAJIME | 5/16 = **0.31** |

e.g. real `@biesterfeld.com` has **four** managers across its 30 labeled users;
the top one (PRANNOY) covers only 10/30 = **0.33**. Because 0.33 < 0.80, the
tiered model does **not** use domain lookup for Biesterfeld — it falls through to
similarity (Approach 2). This is the real reason the Biesterfeld rows in §9 show
`method = similarity` and disagree with one another.

**Real MODERATE majority — "2 of 3" → 0.67 → still gated out**

| email_domain | X1 counts | most common X1 | share |
|---|---|---|---:|
| `hangjiang.com` | CHARLEY HU ×2, CANWEN (BARRY) PENG ×1 | CHARLEY HU | 2/3 = **0.67** |
| `sojitz.com` | MEKHIN ANUWAT ×2, QIANLI XIE ×1 | MEKHIN, ANUWAT | 2/3 = **0.67** |

Approach 1 alone would answer CHARLEY HU at 0.67, but 0.67 < 0.80 → the tiered
model prefers similarity.

**Real SINGLE-user domains — confidence 1.0 by construction (the caveat)**

`3fllc.com → TRICIA COOKE MILLER`, `abbott.com → ANTHONY VERROCCHI`,
`aceway.com.hk → GUOBAO SUN (ASUN)` — each has just **one** labeled user, so share =
1/1 = **1.0**. The number is real but rests on a single example.

**Why this matters — the real shape of the 783 domains**

| | count |
|---|---:|
| distinct email domains among the 783 | **400** |
| domains with ≥ 2 labeled users | 105 |
| domains with exactly 1 labeled user | **295 (74%)** |

So **74% of the domains the rule learns are single-example** (confidence 1.0 "by
construction"), and only a minority have enough users to give a *weighty* vote.
That is precisely why Approach 1 is precise where it has evidence but limited
overall — and why the bigger **enrichment directory** (built from 145k SFDC
contacts, not just 783 users) is the stronger sibling for domain-based mapping.

---

## 4. Approach 2 — Nearest-neighbour similarity (char TF-IDF)

**Idea:** even if a domain is new, find the **most similar labeled user** (by
company + domain text) and borrow their X1.

**Fit:** turn each labeled user's `text` into a character n-gram **TF-IDF vector**
(`analyzer="char_wb", ngram_range=(2,5)`). Character n-grams tolerate spelling
differences (`Acme Inc` ≈ `ACME Incorporated`).

**Predict:** vectorize the unlabeled user's `text`, compute **cosine similarity**
to all 783 labeled vectors, take the closest one, return its X1.
**confidence = the cosine similarity** (1.0 = identical text).

**Strength / weakness:** 100% coverage (always has a nearest neighbour) but lower
precision than the domain rule; handles the long tail.

### 4.1 What the method actually does, step by step

**Step 1 — build the `text` field** (§2.3) for every user:
`text = company + " " + email_domain_name + " " + country` (all lowercased).
Examples:
```
biesterfeld user  →  "biesterfeld biesterfeld germany"
formerra user     →  "formerra formerra canada"
```

**Step 2 — turn text into character n-grams.** `char_wb` means "characters within
word boundaries"; `ngram_range=(2,5)` means every run of 2 to 5 characters.
For the word `omya` the 2–3 character grams are:
```
' o', 'om', 'my', 'ya', 'a ', ' om', 'omy', 'mya', 'ya '
```
Using *characters* (not whole words) is what makes the match tolerant of spelling
and suffix differences — `Acme Inc` and `ACME Incorporated` share many of the same
character grams (`acm`, `cme`, …) even though the words aren't identical.

**Step 3 — TF-IDF weighting.** Each text becomes a numeric vector over all the
character grams seen in the 783 labeled texts. Common grams are down-weighted,
distinctive ones up-weighted, so rare company-specific grams carry more signal.

**Step 4 — cosine similarity.** For an unlabeled user, compare their vector to all
783 labeled vectors. **Cosine similarity** ranges 0 (no shared grams) → 1
(identical text). Take the **single most similar** labeled user and **borrow their
X1**. The similarity value **is** the confidence.

### 4.2 Real nearest-neighbour examples (from the data)

**Example 1 — exact text match → similarity 1.00 (strong, auto-accept)**
```
UNLABELED : a.abbasi@biesterfeld.com
   text    : "biesterfeld biesterfeld germany"
   nearest : "biesterfeld biesterfeld germany"   (m.rathke@biesterfeld.com)
   cosine  : 1.000   → borrow X1 = ANNA BURSAKOVA
```
The unlabeled user's text is *identical* to a labeled user's text, so similarity =
1.0. (Note: Biesterfeld is a multi-manager company, so even at 1.0 the borrowed
manager may not be the *right* one — see the caveat in §9.)

**Example 2 — fuzzy match, same company, different country office → similarity 0.50**
```
UNLABELED : development@kohyei.in
   text    : "polyhose kohyei kohyei india"
   nearest : "kohyei trading co.,ltd kycztj japan"   (yuji_kishi@kycztj.co.jp)
   cosine  : 0.498   → borrow X1 = OHKE, HAJIME
```
No labeled `@kohyei.in` user exists, but the character grams of `kohyei` overlap, so
the nearest neighbour is the Kohyei record in Japan. The match is *partial* (0.50)
because only the `kohyei` part agrees — different country and extra words pull it
down. Confidence 0.50 → borderline → flagged for review.

**Example 3 — spurious look-alike → low similarity (correctly flagged for review)**
```
UNLABELED : nicole@form3.com
   text    : "form3 design form3 canada"
   nearest : "formerra formerra canada"   (alejandro.hernandez@formerra.com)
   cosine  : 0.499   → borrow X1 = AMINE OURABAH
```
Here `form3` and `formerra` share the grams `for`, `orm`, `form`, plus both are in
Canada — so they look similar to the character matcher, but they are **different
companies**. The model returns a low confidence (0.499), so `needs_review = True`
and a human catches it. This is exactly why the **confidence threshold** matters:
weak, possibly-wrong matches surface as low scores instead of silent errors.

> Reading the three together: similarity is **highest when the whole text matches**
> (Example 1), **medium when only the company name matches** (Example 2), and **low
> when only a few characters coincidentally overlap** (Example 3). The confidence
> score tracks that, which is what lets the tiered model auto-accept the strong ones
> and route the weak ones to a person.

---

## 5. The tiered model (what we actually ship)

Combine the two so each does what it is best at:

```
Tier 1 — domain lookup, used ONLY when it is confident (vote share ≥ 0.80)
           i.e. the domain maps cleanly to one manager
Tier 2 — similarity (char-TFIDF) for everyone else
```

Why the 0.80 gate: if a domain's labeled users are split across managers, the
majority vote is unreliable, so we let the more precise similarity match decide.

---

## 6. How we measured accuracy (held-out test)

We cannot check the 4,071 directly (no true answer). So we **hold out 25% of the
783** as a test set, train on the other 75%, predict the held-out users, and
compare to their known X1.

> Caveat printed in the notebook: with ~4 examples per manager this is
> *indicative*, not a tight estimate — many managers have too few examples to
> appear in both train and test.

**Results (196 test / 587 train):**

| Method | Coverage | Accuracy (on covered) |
|--------|---------:|----------------------:|
| 1. Domain lookup | 64.8% | **74.0%** |
| 2. Char-TFIDF NN | 100% | 58.2% |
| 3. Tiered (1→2) | 100% | **59.2%** ← best overall |

---

## 7. Confidence is a reliable triage signal

The tiered model's accuracy rises sharply with its confidence score — this is
what enables the **human-in-the-loop** design (auto-accept high confidence, send
the rest to a person):

| Min confidence to auto-accept | Share auto-accepted | Accuracy |
|------------------------------:|--------------------:|---------:|
| ≥ 0.5 | 72% | 77% |
| ≥ 0.8 | 60% | **88%** |
| ≥ 0.9 | 47% | **89%** |

So we can auto-accept ~60% of predictions at ~88% accuracy and review the rest.

---

## 8. Approach 4 — Embedding / LLM semantic match (tested, not adopted)

**Idea:** replace char-TF-IDF with a sentence-transformer (`all-MiniLM-L6-v2`)
that encodes `text` into a 384-dim "meaning" vector; nearest neighbour in that
space; cosine = confidence.

**Result (same held-out split):**

| Method | Accuracy |
|--------|---------:|
| Char-TFIDF NN (Approach 2) | **58.2%** |
| Embedding NN (Approach 4) | 56.6% ← slightly worse |
| Tiered (embedding) | 59.2% (tie with TFIDF) |

**Conclusion: embeddings did NOT help.** X1 is tied to an *exact* company, so
semantic generalization (matching different-but-similar companies) adds noise.
Embedding confidence is also **over-confident** (short company strings all embed
close together), making its review threshold less useful. We therefore keep the
simpler char-TF-IDF tiered model: equal/better accuracy, better-calibrated
confidence, no model download, far faster.

---

## 9. Applying the shipped model to the 4,071 unlabeled users

Fit the tiered model on **all 783** labels, predict the 4,071, write
`chemille_user_x1_phase2_predictions.csv`.

**Breakdown:**
- by method: **814** via domain lookup, **3,257** via similarity
- **1,809** auto-accept (confidence ≥ 0.5) · **2,262** flagged for human review

**Real predictions — high confidence (auto-accept):**
```
chemille_email                company                     predicted_X1            conf  method
9001093@parker.com            Parker Hannifin Japan Ltd.  TIMOTHY BUTURLA         1.0   domain_lookup
a.yasser@snetor.com           Snetor                      PRANNOY VINCENT ALVA    1.0   domain_lookup
a.abbasi@biesterfeld.com      Biesterfeld                 ANNA BURSAKOVA          1.0   similarity
a.arndt@biesterfeld.com       Biesterfeld                 JOSE GONZALEZ           1.0   similarity
```

**Real predictions — low confidence (needs review):**
```
chemille_email                company                              predicted_X1        conf   method
hausberger@id-design.de       ID Design Produktentwicklung GmbH    MARYJANE ARMISTEAD  0.499  similarity
nicole@form3.com              Form3 Design                         AMINE OURABAH       0.499  similarity
development@kohyei.in         Polyhose Kohyei                      OHKE, HAJIME        0.498  similarity
```

> Note the Biesterfeld rows: same company, **different** predicted managers — the
> similarity match picks the nearest labeled Biesterfeld user, but because that
> company has several managers the choice can be arbitrary. This multi-manager
> ambiguity is the main known weakness and needs an extra signal (region/segment)
> to resolve.

---

## 10. Summary

| | |
|---|---|
| Train on | 783 labeled users |
| Predict | 4,071 unlabeled users |
| Inputs | `email_domain`, `country`, `company` (Chemille fields only) |
| Output | `predicted_X1` + `confidence` + `method` + `needs_review` |
| Best model | tiered: domain lookup (≥0.80) → char-TFIDF similarity |
| Held-out accuracy | 59% overall; **~88% at confidence ≥ 0.8** |
| Embeddings (Approach 4) | tested, no improvement → not adopted |
| Deployment | auto-accept high confidence, route low confidence to a human |

**One paragraph:** Phase 2 trains on the 783 known users using only the three
Chemille-file inputs (`email_domain`, `country`, `company`) and predicts the X1
manager for the 4,071 unknowns. A tiered model — confident domain lookup first,
char-TF-IDF nearest-neighbour otherwise — scores ~59% overall but ~88% on the
high-confidence ~60%, enabling auto-accept-plus-human-review. A sentence-embedding
variant was tested and gave no improvement, so the simpler, faster, better-
calibrated TF-IDF model is the one we ship.
