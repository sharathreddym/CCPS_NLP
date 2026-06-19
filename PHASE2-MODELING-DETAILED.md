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
