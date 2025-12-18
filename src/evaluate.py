from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    print(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds, labels=["high_risk", "medium_risk", "low_risk"])
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["high", "medium", "low"],
                yticklabels=["high", "medium", "low"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
