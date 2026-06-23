# Chemille → X1 Account Manager — Final Summary

**Goal:** assign an X1 Account Manager to every Chemille portal user.

## Result

Of **4,854 external Chemille users** (after removing 1,447 internal `@celanese.com`):

| | Users | % |
|---|---:|---:|
| ✅ Confirmed (exact lookup + enrichment) | **1,510** | 31% |
| 🔵 Predicted (model) | 3,344 | 69% |
| **Auto-usable** (confirmed + high-confidence) | **3,049** | **63%** |
| 👁️ Flagged for human review | **1,805** | **37%** |

Blended accuracy of the prediction model (held-out test): **~82%**.

## The single result file — `chemille_users_x1_FINAL_all.csv`

One row per Chemille user, with their X1 manager and how it was obtained:

| Column | Meaning |
|---|---|
| `chemille_email`, `company`, `country` | the user |
| `X1_Account_Manager` | the assigned manager |
| `source` | how it was found (see below) |
| `confidence` | 0–1 |
| `needs_review` | `True` = please check before using |

**`source` values & counts:**
| source | users | trust |
|---|---:|---|
| Phase1 - exact email (confirmed) | 783 | certain |
| Enrichment - email domain (confirmed) | 727 | 96% |
| Predicted - domain lookup | 440 | ~82% |
| Predicted - account name match | 492 | ~81% |
| Predicted - text similarity | 2,412 | ~38% (most flagged) |

**How to use it:** apply directly where `needs_review = False` (3,049 users); send
the `needs_review = True` rows (1,805) to the commercial team for a quick check.

## How it works — `approach_diagram.png`

A 4-tier waterfall; each user takes the **first** method that is confident:
1. **Enrichment** — match a colleague at the same company (email domain) to the real SAP record.
2. **Domain lookup** — same domain as known users → their manager.
3. **Account-name match** — fuzzy-match the company name to SFDC accounts.
4. **Text similarity** — nearest known company by name (fallback).

## The model — `final_model_pipeline.ipynb` (`.py`)

Runs the full waterfall and produces the predictions. Reads the four source files
(Chemille, SFDC Contacts, SFDC Accounts, SAP). The last cell, `predict_final_one(...)`,
lets you predict the X1 for any single new user.

---

*Folder contents: this summary · the result CSV · the model notebook/script · the approach diagram.*
