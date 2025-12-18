from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_models(X, y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
    }

    pipelines = {}

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", model)
        ])
        pipeline.fit(X_train, y_train)
        pipelines[name] = pipeline

    return pipelines, X_test, y_test
