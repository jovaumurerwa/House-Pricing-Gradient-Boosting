import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from house_prices.features import ID, build_features

SEED = 42
MAX_CSV_BYTES = 50 * 1024 * 1024
XGB_PARAMS = dict(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=SEED,
)


def read_csv_safely(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. See README: 'Get the data'.")
    if path.stat().st_size > MAX_CSV_BYTES:
        raise ValueError(f"{path.name} is unexpectedly large. Refusing to load it.")
    return pd.read_csv(path)


def cv_rmse(model, X, y) -> tuple[float, float]:
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = -cross_val_score(model, X, y, cv=folds, scoring="neg_root_mean_squared_error")
    return float(scores.mean()), float(scores.std())


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    from xgboost import XGBRegressor

    parser = argparse.ArgumentParser(description="Train the house price model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    train = read_csv_safely(args.data_dir / "train.csv")
    test = read_csv_safely(args.data_dir / "test.csv")
    X, X_test, y_log = build_features(train, test)

    models = {
        "Ridge regression (baseline)": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "XGBoost": XGBRegressor(**XGB_PARAMS),
    }
    print("5-fold cross-validated RMSE (log scale, lower is better)")
    for name, model in models.items():
        mean, std = cv_rmse(model, X.astype(float), y_log)
        print(f"  {name:<28} {mean:.4f} ± {std:.4f}")

    final_model = XGBRegressor(**XGB_PARAMS)
    final_model.fit(X.astype(float), y_log)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "house_price_xgb.json"
    final_model.save_model(model_path)
    (args.out_dir / "house_price_xgb.json.sha256").write_text(sha256_of(model_path) + "\n")

    predictions = np.expm1(final_model.predict(X_test.astype(float)))
    pd.DataFrame({ID: test[ID], "SalePrice": predictions}).to_csv(args.out_dir / "submission.csv", index=False)
    print(f"Saved model to {model_path} (JSON, not pickle) and predictions to {args.out_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()
