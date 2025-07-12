import joblib, json, argparse, numpy as np
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / 'models'

def load_artifacts(model_name):
    model_file = MODEL_DIR / f"{model_name}.pkl"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file {model_file} not found. Run train.py first.")
    model = joblib.load(model_file)
    with open(MODEL_DIR / 'symptom_index.json') as f:
        symptom_index = json.load(f)
    return model, symptom_index

def vectorize(symptoms, symptom_index):
    x = np.zeros(len(symptom_index), dtype=int)
    for s in symptoms:
        key = s.strip().lower().replace(' ', '_')
        # Each column was originally named like 'itching', 'skin_rash'
        # Accept either 'itching' or 'Itching' etc.
        if key in symptom_index:
            x[symptom_index[key]] = 1
    return x.reshape(1, -1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['decision_tree', 'naive_bayes', 'tree', 'nb'], default='naive_bayes')
    args = parser.parse_args()
    model_alias = {'tree': 'decision_tree', 'nb': 'naive_bayes'}
    model_name = model_alias.get(args.model, args.model)

    model, symptom_index = load_artifacts(model_name)

    user_input = input("Enter comma‑separated symptoms: ")
    symptoms = [s.strip() for s in user_input.split(',') if s.strip()]
    x = vectorize(symptoms, symptom_index)
    prediction = model.predict(x)[0]
    print(f"\nPredicted disease: {prediction}")

if __name__ == '__main__':
    main()