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


