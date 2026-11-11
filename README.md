# House Price Prediction with Gradient Boosting

Predicts house sale prices from 79 property features using **XGBoost**, with careful handling of missing data, ordinal encoding of quality ratings, and a log-transformed target. Includes a reproducible training pipeline, a baseline comparison, tests, and CI security scans.

Dataset: [Kaggle – House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) (1,460 training homes, 1,459 test homes).

## Why this matters

Pricing property fairly is hard when information is incomplete. In growing cities like Kigali and Nairobi, buyers, lenders and planners need quick, explainable estimates. This project shows the full process on a well-known dataset, so the same approach can be applied to local property data.

## Approach

1. **Target:** sale prices are right-skewed, so the model learns `log(1 + price)`. This stops expensive houses from dominating the error.
2. **Missing values, filled by meaning:**
   - "No pool / no garage / no basement" → `None` or `0` (missing means *absent*, not *unknown*)
   - Lot frontage → median of the same neighbourhood
   - A few categorical gaps → most common value
3. **Encoding:**
   - Quality ratings (Poor → Excellent) mapped to 0–5, so the model keeps their order
   - `MSSubClass` treated as a category, not a number
   - Remaining categories one-hot encoded (228 features in total)
4. **Model:** XGBoost (800 shallow trees, learning rate 0.03, row and column subsampling to reduce overfitting).
5. **Evaluation:** RMSE on the log scale, compared against a Ridge regression baseline with 5-fold cross-validation.

## Results

| Model | RMSE (log scale) |
|---|---|
| XGBoost, 80/20 hold-out (notebook) | **0.1319** |


An RMSE of 0.13 on the log scale means predictions are typically within about **13–14%** of the real price.

## Security and good practice

| Risk | What this project does |
|---|---|
| Leaking the Kaggle API key | Key stays in `~/.kaggle/kaggle.json` with `chmod 600`; `kaggle.json` is git-ignored |
| Redistributing licensed data | Data is downloaded by script and git-ignored, never committed |
| Loading a model file that runs hidden code | Model saved as **XGBoost JSON**, not pickle, plus a SHA-256 fingerprint |
| Malformed or unexpected input files | Required columns are checked; oversized files are refused |
| Vulnerable libraries | Only 5 pinned libraries; `pip-audit` runs in CI |
| Unsafe code / leaked secrets | Bandit and Gitleaks run in CI on every push |

## Run it

```bash
git clone https://github.com/jovaumurerwa/House-Pricing-Gradient-Boosting.git
cd House-Pricing-Gradient-Boosting
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Get the data** (needs a free Kaggle account and API key):

```bash
pip install kaggle
./scripts/download_data.sh
```

**Train and evaluate:**

```bash
python -m house_prices.train
```

This prints the cross-validation comparison and writes `outputs/house_price_xgb.json` and `outputs/submission.csv`.

**Explore:** open `house_prices.ipynb` for the step-by-step analysis and charts.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

The tests check that no values are left missing, that train and test end up with the same columns, that ratings become ordered numbers, that IDs are never used as features, and that bad input files are rejected.

## Project structure

```
house_prices.ipynb         Exploration, charts and first results
house_prices/features.py   Cleaning and encoding (shared, tested)
house_prices/train.py      Cross-validation, final training, safe model saving
scripts/download_data.sh   Downloads the dataset with the Kaggle API
tests/                     Pytest suite
```

## Limitations and next steps

- Missing values are filled using train and test together. This does not leak prices, but a stricter version would learn the fill values from training data only.
- Next: tune hyperparameters with cross-validation, try LightGBM, and explain individual predictions with SHAP.
- Apply the pipeline to real Rwandan or Kenyan property listings.

## What I learned

- Reading the data dictionary matters: "missing" often means "the house doesn't have it".
- Transforming a skewed target can improve a model more than tuning it.
- A notebook is good for exploring; a tested script is what makes results reproducible.
