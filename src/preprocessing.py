import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(path):
    return pd.read_csv(path, sep=';')

def preprocess(df):
    X = df.drop("G3", axis=1)
    y = df["G3"]

    # Convert final grade to risk classes
    y = pd.cut(
        y,
        bins=[-1, 9, 13, 20],
        labels=["high_risk", "medium_risk", "low_risk"]
    )

    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(exclude="object").columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols)
        ]
    )

    return X, y, preprocessor
