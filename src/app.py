import json
import joblib
import pathlib
import streamlit as st

# Locate the models folder relative to this file

BASE_DIR  = pathlib.Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"


# Load trained model + symptom index

model           = joblib.load(MODEL_DIR / "naive_bayes.pkl")
symptom_index   = json.loads((MODEL_DIR / "symptom_index.json").read_text())
symptoms_list   = list(symptom_index.keys())          # for dropdown

# Disease → {cure, doctor}.  Add more entries freely.

disease_info = {
    "(vertigo) Paroymsal  Positional Vertigo": {
        "cure": "Vestibular therapy and head movement exercises.",
        "doctor": "Neurologist"
    },
    "AIDS": {
        "cure": "Antiretroviral therapy (ART) and immune support.",
        "doctor": "Infectious Disease Specialist"
    },
    "Acne": {
        "cure": "Topical retinoids and proper skincare.",
        "doctor": "Dermatologist"
    },
    "Alcoholic hepatitis": {
        "cure": "Stop alcohol, corticosteroids, and nutritional therapy.",
        "doctor": "Hepatologist"
    },
    "Allergy": {
        "cure": "Antihistamines and allergen avoidance.",
        "doctor": "Allergist"
    },
    "Arthritis": {
        "cure": "Anti-inflammatory drugs and physical therapy.",
        "doctor": "Rheumatologist"
    },
    "Bronchial Asthma": {
        "cure": "Inhaled bronchodilators and avoid allergens.",
        "doctor": "Pulmonologist"
    },
    "Cervical spondylosis": {
        "cure": "Physiotherapy and pain management.",
        "doctor": "Orthopedic Specialist"
    },
    "Chicken pox": {
        "cure": "Antiviral medication and rest.",
        "doctor": "General Physician"
    },
    "Chronic cholestasis": {
        "cure": "Bile acid therapy and symptom control.",
        "doctor": "Gastroenterologist"
    },
    "Common Cold": {
        "cure": "Rest, hydration, and over-the-counter meds.",
        "doctor": "General Physician"
    },
    "Dengue": {
        "cure": "Hydration and pain relievers. No specific cure.",
        "doctor": "Infectious Disease Specialist"
    },
    "Diabetes ": {
        "cure": "Insulin, diet control, and regular monitoring.",
        "doctor": "Endocrinologist"
    },
    "Dimorphic hemmorhoids(piles)": {
        "cure": "Fiber-rich diet, sitz baths, and surgery if needed.",
        "doctor": "General Surgeon"
    },
    "Drug Reaction": {
        "cure": "Discontinue drug and use antihistamines or steroids.",
        "doctor": "Allergist"
    },
    "Fungal infection": {
        "cure": "Antifungal creams or oral antifungals.",
        "doctor": "Dermatologist"
    },
    "GERD": {
        "cure": "Avoid acidic foods, use antacids, lifestyle changes.",
        "doctor": "Gastroenterologist"
    },
    "Gastroenteritis": {
        "cure": "Oral rehydration and rest. Antibiotics if bacterial.",
        "doctor": "Gastroenterologist"
    },
    "Heart attack": {
        "cure": "Emergency medical treatment and surgery if needed.",
        "doctor": "Cardiologist"
    },
    "Hepatitis B": {
        "cure": "Antiviral medication and liver monitoring.",
        "doctor": "Hepatologist"
    },
    "Hepatitis C": {
        "cure": "Direct-acting antivirals (DAAs).",
        "doctor": "Hepatologist"
    },
    "Hepatitis D": {
        "cure": "Interferon alpha therapy and liver care.",
        "doctor": "Hepatologist"
    },
    "Hepatitis E": {
        "cure": "Rest, fluids, and avoid alcohol.",
        "doctor": "Hepatologist"
    },
    "Hypertension ": {
        "cure": "Lifestyle changes and antihypertensive drugs.",
        "doctor": "Cardiologist"
    },
    "Hyperthyroidism": {
        "cure": "Antithyroid drugs or radioactive iodine.",
        "doctor": "Endocrinologist"
    },
    "Hypoglycemia": {
        "cure": "Consume fast-acting carbs. Monitor sugar levels.",
        "doctor": "Endocrinologist"
    },
    "Hypothyroidism": {
        "cure": "Thyroid hormone replacement therapy.",
        "doctor": "Endocrinologist"
    },
    "Impetigo": {
        "cure": "Topical or oral antibiotics.",
        "doctor": "Dermatologist"
    },
    "Jaundice": {
        "cure": "Treat underlying cause. Hydration and diet.",
        "doctor": "Gastroenterologist"
    },
    "Malaria": {
        "cure": "Antimalarial drugs like chloroquine or artemisinin.",
        "doctor": "General Physician"
    },
    "Migraine": {
        "cure": "Pain relievers and lifestyle adjustments.",
        "doctor": "Neurologist"
    },
    "Osteoarthristis": {
        "cure": "Exercise, weight control, joint injections.",
        "doctor": "Orthopedic Specialist"
    },
    "Paralysis (brain hemorrhage)": {
        "cure": "Rehabilitation and physical therapy.",
        "doctor": "Neurologist"
    },
    "Peptic ulcer diseae": {
        "cure": "Proton pump inhibitors and avoid spicy foods.",
        "doctor": "Gastroenterologist"
    },
    "Pneumonia": {
        "cure": "Antibiotics, antivirals, and rest.",
        "doctor": "Pulmonologist"
    },
    "Psoriasis": {
        "cure": "Topical treatments, phototherapy, biologics.",
        "doctor": "Dermatologist"
    },
    "Tuberculosis": {
        "cure": "Long-term antibiotics (6-9 months).",
        "doctor": "Pulmonologist"
    },
    "Typhoid": {
        "cure": "Antibiotics and fluid replacement.",
        "doctor": "General Physician"
    },
    "Urinary tract infection": {
        "cure": "Antibiotics and increased fluid intake.",
        "doctor": "Urologist"
    },
    "Varicose veins": {
        "cure": "Compression stockings or vein surgery.",
        "doctor": "Vascular Surgeon"
    },
    "hepatitis A": {
        "cure": "Rest, fluids, and avoid alcohol.",
        "doctor": "Hepatologist"
    }
}

# Streamlit UI

st.set_page_config(page_title="Disease Prediction", layout="centered")
st.title("🧠 Disease Prediction from Symptoms")
st.markdown("Select the symptoms you are experiencing:")

selected = st.multiselect("Symptoms", options=symptoms_list)

if st.button("Predict Disease") and selected:
    # Build binary feature vector
    x = [0] * len(symptom_index)
    for s in selected:
        x[symptom_index[s]] = 1

    diagnosis = model.predict([x])[0]

    st.success(f"**Predicted Disease:** {diagnosis}")

    info = disease_info.get(diagnosis)
    if info:
        st.info(f"**Cure/Treatment:** {info['cure']}")
        st.info(f"**Recommended Doctor:** {info['doctor']}")
    else:
        st.warning("No cure/doctor information available for this disease. Please consult a physician.")


