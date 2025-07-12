import os, json, joblib, argparse
import pandas as pd
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report

DATA_DIR  = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True)

def load_data():
    train_df = pd.read_csv(DATA_DIR / "Training.csv")
    test_df  = pd.read_csv(DATA_DIR / "Testing.csv")

    # drop empty trailing column if present
    for df in (train_df, test_df):
        df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], inplace=True, errors="ignore")

    # turn every NaN into 0 (symptom absent)
    train_df.fillna(0, inplace=True)
    test_df.fillna(0, inplace=True)

    return train_df, test_df

def preprocess(df):
    X = df.drop("prognosis", axis=1)
    y = df["prognosis"]
    return X, y

def main():
    train_df, test_df = load_data()
    X_train, y_train = preprocess(train_df)
    X_test,  y_test  = preprocess(test_df)

    # confirm no NaNs (debug)
    assert X_train.isnull().sum().sum() == 0, "NaNs still in X_train"
    assert X_test.isnull().sum().sum()  == 0, "NaNs still in X_test"

    # build models
    tree_clf = DecisionTreeClassifier(random_state=42)

    nb_clf = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0),  # guarantees no NaN
        BernoulliNB()
    )

    # train
    tree_clf.fit(X_train, y_train)
    nb_clf.fit(X_train, y_train)

    # evaluate
    for name, model in [("Decision Tree", tree_clf), ("Bernoulli NB", nb_clf)]:
        y_pred = model.predict(X_test)
        print(f"{name} accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred)[:400], "\n")

    # save
    joblib.dump(tree_clf, MODEL_DIR / "decision_tree.pkl")
    joblib.dump(nb_clf,   MODEL_DIR / "naive_bayes.pkl")

    # save symptom index
    symptom_index = {sym: idx for idx, sym in enumerate(X_train.columns)}
    with open(MODEL_DIR / "symptom_index.json", "w") as f:
        json.dump(symptom_index, f, indent=2)

    print("✅ Models and symptom_index.json saved to", MODEL_DIR)

if __name__ == "__main__":
    main()
