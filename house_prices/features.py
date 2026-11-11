import numpy as np
import pandas as pd

TARGET = "SalePrice"
ID = "Id"

FILL_WITH_NONE = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]
FILL_WITH_ZERO = [
    "GarageYrBlt", "GarageArea", "GarageCars", "MasVnrArea",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath",
]
FILL_WITH_MODE = [
    "MSZoning", "Utilities", "Exterior1st", "Exterior2nd",
    "Electrical", "KitchenQual", "Functional", "SaleType",
]

QUALITY_SCALE = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
QUALITY_COLUMNS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]
BASEMENT_FINISH = {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
ORDINAL_MAPS = {
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": BASEMENT_FINISH,
    "BsmtFinType2": BASEMENT_FINISH,
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
}

REQUIRED_COLUMNS = set(
    FILL_WITH_NONE + FILL_WITH_ZERO + FILL_WITH_MODE + QUALITY_COLUMNS
    + ["LotFrontage", "Neighborhood", "MSSubClass", ID]
)


class InvalidDataError(ValueError):
    """Raised when an input file is missing columns the pipeline needs."""


def validate_columns(frame: pd.DataFrame, name: str, needs_target: bool) -> None:
    required = REQUIRED_COLUMNS | ({TARGET} if needs_target else set())
    missing = sorted(required - set(frame.columns))
    if missing:
        raise InvalidDataError(f"{name} is missing columns: {', '.join(missing[:10])}")


def fill_missing(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data[FILL_WITH_NONE] = data[FILL_WITH_NONE].fillna("None")
    data[FILL_WITH_ZERO] = data[FILL_WITH_ZERO].fillna(0)
    data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
    data["LotFrontage"] = data["LotFrontage"].fillna(data["LotFrontage"].median())
    for column in FILL_WITH_MODE:
        data[column] = data[column].fillna(data[column].mode()[0])
    return data


def encode(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["MSSubClass"] = data["MSSubClass"].astype(str)
    for column in QUALITY_COLUMNS:
        data[column] = data[column].map(QUALITY_SCALE)
    for column, mapping in ORDINAL_MAPS.items():
        data[column] = data[column].map(mapping)
    return pd.get_dummies(data, drop_first=True)


def build_features(train: pd.DataFrame, test: pd.DataFrame):
    """Return (X_train, X_test, y_log) with identical columns for train and test."""
    validate_columns(train, "train.csv", needs_target=True)
    validate_columns(test, "test.csv", needs_target=False)

    y_log = np.log1p(train[TARGET])
    combined = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)
    combined = encode(fill_missing(combined)).fillna(0)

    features = combined.drop(columns=[ID])
    n_train = len(train)
    return features.iloc[:n_train], features.iloc[n_train:], y_log
