import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder

model = tf.keras.models.load_model('dementia_prediction_model.h5')

def preprocess_input(data):
    data = pd.DataFrame([data])

    label_encoders = {
        "Prescription": LabelEncoder().fit(['None', 'Galantamine', 'Donepezil', 'Memantine', 'Rivastigmine']),
        "Education_Level": LabelEncoder().fit(['Primary School', 'Secondary School', 'No School', 'Diploma/Degree']),
        "Dominant_Hand": LabelEncoder().fit(['Left', 'Right']),
        "Gender": LabelEncoder().fit(['Female', 'Male']),
        "Family_History": LabelEncoder().fit(['No', 'Yes']),
        "Smoking_Status": LabelEncoder().fit(['Never Smoked', 'Former Smoker', 'Current Smoker']),
        "APOE_ε4": LabelEncoder().fit(['Negative', 'Positive']),
        "Physical_Activity": LabelEncoder().fit(['Sedentary', 'Mild Activity', 'Moderate Activity']),
        "Depression_Status": LabelEncoder().fit(['No', 'Yes']),
        "Nutrition_Diet": LabelEncoder().fit(['Balanced Diet', 'Mediterranean Diet', 'Low-Carb Diet']),
        "Sleep_Quality": LabelEncoder().fit(['Poor', 'Good']),
        "Chronic_Health_Conditions": LabelEncoder().fit(['None', 'Diabetes', 'Heart Disease', 'Hypertension']),
        "Medication_History": LabelEncoder().fit(['No', 'Yes']),  
    }

    for key, encoder in label_encoders.items():
        if key in data.columns:
            try:
                data[key] = data[key].astype(str)  
                data[key] = encoder.transform(data[key])
            except ValueError as e:
                raise ValueError(f"Error encoding column '{key}': {e}")

    numerical_columns = [
        "Diabetic", "AlcoholLevel", "HeartRate", "BloodOxygenLevel",
        "BodyTemperature", "Weight", "MRI_Delay", "Dosage in mg",
        "Age", "Cognitive_Test_Scores"
    ]
    
    for col in numerical_columns:
        if col in data.columns:
            try:
                data[col] = pd.to_numeric(data[col], errors='coerce')  # บังคับแปลงเป็นตัวเลข
            except Exception as e:
                raise ValueError(f"Error converting column '{col}' to numeric: {e}")

    if data.isnull().values.any():
        data = data.fillna(data.mean())

    try:
        scaler = StandardScaler()
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    except Exception as e:
        raise ValueError(f"Error during scaling: {e}")

    invalid_dtypes = data.select_dtypes(include=['object'])
    if not invalid_dtypes.empty:
        raise ValueError(f"Invalid dtype columns remaining: {invalid_dtypes.columns.tolist()}")

    return data.values




st.title("Dementia Prediction App")
st.write("กรอกข้อมูลด้านล่างเพื่อพยากรณ์ว่ามีความเสี่ยงเป็นโรงสมองเสื่อมหรือไม่")

input_data = {
    "Diabetic": st.selectbox("Diabetic (เบาหวาน) (1 คือ เป็น, 0 คือ ไม่เป็น)", [0, 1]),
    "AlcoholLevel": st.slider("Alcohol Level", 0.0, 1.0, 0.15),
    "HeartRate": st.number_input("Heart Rate (อัตราการเต้นของหัวใจ)", value=67),
    "BloodOxygenLevel": st.number_input("Blood Oxygen Level (ปริมาณออกซิเจนในเลือด)", value=97.5),
    "BodyTemperature": st.number_input("Body Temperature (อุณหภูมิร่างกาย)", value=36.0),
    "Weight": st.number_input("Weight (น้ำหนัก)", value=68.0),
    "MRI_Delay": st.number_input("MRI Delay (วัน)", value=28.0),
    "Prescription": st.selectbox("Prescription (ยาที่ใช้)", ['None', 'Galantamine', 'Donepezil', 'Memantine', 'Rivastigmine']),
    "Dosage in mg": st.number_input("Dosage (ปริมาณยา)", value=20.0),
    "Age": st.number_input("Age (อายุ)", value=77),
    "Education_Level": st.selectbox("Education Level", ['Primary School', 'Secondary School', 'No School', 'Diploma/Degree']),
    "Dominant_Hand": st.selectbox("Dominant Hand (ถนัดมือ)", ['Left', 'Right']),
    "Gender": st.selectbox("Gender (เพศ)", ['Female', 'Male']),
    "Family_History": st.selectbox("Family History (ประวัติครอบครัว)", ['No', 'Yes']),
    "Smoking_Status": st.selectbox("Smoking Status (สถานะการสูบบุหรี่)", ['Never Smoked', 'Former Smoker', 'Current Smoker']),
    "APOE_ε4": st.selectbox("APOE_ε4 (ผลตรวจ)", ['Negative', 'Positive']),
    "Physical_Activity": st.selectbox("Physical Activity (กิจกรรมทางกาย)", ['Sedentary', 'Mild Activity', 'Moderate Activity']),
    "Depression_Status": st.selectbox("Depression Status (ซึมเศร้า)", ['No', 'Yes']),
    "Cognitive_Test_Scores": st.number_input("Cognitive Test Scores (คะแนนการทดสอบการรับรู้)", value=0),
    "Medication_History": st.selectbox("Medication History (ประวัติการใช้ยา)", ['No', 'Yes']),
    "Nutrition_Diet": st.selectbox("Nutrition Diet (อาหาร)", ['Balanced Diet', 'Mediterranean Diet', 'Low-Carb Diet']),
    "Sleep_Quality": st.selectbox("Sleep Quality (คุณภาพการนอน)", ['Poor', 'Good']),
    "Chronic_Health_Conditions": st.selectbox("Chronic Health Conditions (ปัญหาสุขภาพเรื้อรัง)", ['None', 'Diabetes', 'Heart Disease', 'Hypertension']),
}

if st.button("Predict"):
    try:
        processed_data = preprocess_input(input_data)
    
        
        prediction = model.predict(processed_data)[0][0]
        
        if prediction > 0.5:
            st.error("ผลลัพธ์: มีความเสี่ยงเป็นโรคสมองเสื่อม")
        else:
            st.success("ผลลัพธ์: ไม่มีความเสี่ยงเป็นโรคสมองเสื่อม")
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")
