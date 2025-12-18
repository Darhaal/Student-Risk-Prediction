from src.preprocessing import load_data, preprocess
from src.features import add_features
from src.train import train_models
from src.evaluate import evaluate

def main():
    df = load_data("data/student-mat.csv")
    df = add_features(df)

    X, y, preprocessor = preprocess(df)
    models, X_test, y_test = train_models(X, y, preprocessor)

    for name, model in models.items():
        print(f"\n=== {name} ===")
        evaluate(model, X_test, y_test)

if __name__ == "__main__":
    main()
