<<<<<<< HEAD

# Disease Prediction from Symptoms

This project builds two machine‑learning models (Decision Tree & Naive Bayes) that predict the **most probable disease** given a set of symptoms.

## 1. Dataset

Download the open dataset that contains 132 binary symptom columns and a `prognosis` column. A popular copy is available on Kaggle (e.g. *Disease Prediction Using Machine Learning*). Place:

```
Training.csv
Testing.csv
```

inside **`./data/`**.

## 2. Setup

```bash
python -m venv venv
source venv/bin/activate         # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Train the models

```bash
python src/train.py
```

This will:
* Train a Decision Tree and Bernoulli Naive Bayes on `Training.csv`
* Evaluate on `Testing.csv`
* Save the fitted artefacts in **`./models/`**

## 4. Run interactive prediction

```bash
python src/predict.py  --model nb   # or --model tree
```

You will be prompted to type symptoms separated by commas:

```
Enter comma‑separated symptoms: itching,skin rash,nodal skin eruptions
```

The script returns the predicted disease.

## 5. Quick Streamlit demo (optional)

```bash
streamlit run src/app.py
```

## Folder structure

```
disease_prediction_project/
├── data/                # put Training.csv & Testing.csv here
├── models/              # generated after training
├── src/
│   ├── train.py
│   ├── predict.py
│   └── app.py
├── README.md
└── requirements.txt
```

Enjoy!
=======
# 🧠 Disease Prediction from Symptoms

A machine learning-powered web application that predicts possible diseases based on selected symptoms. The app also provides recommended cures and the type of doctor to consult. Built using **Python**, **Streamlit**, and **scikit-learn**.

---

## 🚀 Features

- 🔍 Predicts disease based on selected symptoms
- 💊 Suggests possible cures and treatments
- 👨‍⚕️ Recommends the appropriate doctor to consult
- 📈 Trained on real-world symptom-disease mapping
- 🧠 Machine Learning model (Naive Bayes Classifier)
- 🖥️ Built with Streamlit for an interactive user interface

---

## 🏗️ Project Structure

disease_prediction_project/
│
├── data/
│ ├── Training.csv # Training dataset
│ └── Testing.csv # Testing dataset
│
├── models/
│ ├── naive_bayes.pkl # Trained Naive Bayes model
│ └── symptom_index.json # JSON mapping of symptoms to indexes
│
├── src/
│ ├── train.py # Model training script
│ └── app.py # Streamlit web application
│
├── .gitignore
├── requirements.txt

---
## 📊 Dataset
- The model is trained on a labeled dataset with 132 symptoms mapped to 41 diseases.
- Source: Curated from open-source medical datasets


>>>>>>> 922a27e536f80ea385581185dc5f7fb34f91b758
