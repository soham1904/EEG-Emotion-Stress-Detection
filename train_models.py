import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import streamlit as st


stress_data = pd.read_csv('stress.csv')
emotion_data = pd.read_csv('heart_rate_emotion_dataset.csv')

print("Dataset Overview:")
print(stress_data.head())

X_stress = stress_data.drop(columns=['Emotion', 'Subject'], errors='ignore')
y_stress = stress_data['Emotion']

X_stress_train, X_stress_test, y_stress_train, y_stress_test = train_test_split(
    X_stress, y_stress, test_size=0.2, random_state=42
)

stress_scaler = StandardScaler()
X_stress_train_scaled = stress_scaler.fit_transform(X_stress_train)
X_stress_test_scaled = stress_scaler.transform(X_stress_test)
stress_data = stress_data.dropna(subset=['Emotion'])


stress_model = RandomForestClassifier(n_estimators=100, random_state=42)
stress_model.fit(X_stress_train_scaled, y_stress_train)

y_stress_pred = stress_model.predict(X_stress_test_scaled)
accuracy = accuracy_score(y_stress_test, y_stress_pred)
report = classification_report(y_stress_test, y_stress_pred, target_names=['Relaxed', 'Stressed', 'Neutral'])

print("\nModel Evaluation:")
print("Accuracy:", accuracy)
print("Classification Report:\n", report)

with open('model/stress_model.pkl', 'wb') as model_file:
    pickle.dump(stress_model, model_file)

with open('model/stress_scaler.pkl', 'wb') as scaler_file:
    pickle.dump(stress_scaler, scaler_file)

X_emotion = emotion_data.drop(columns=['Emotion'])
y_emotion = emotion_data['Emotion']
X_emotion_train, X_emotion_test, y_emotion_train, y_emotion_test = train_test_split(X_emotion, y_emotion, test_size=0.2, random_state=42)

emotion_scaler = StandardScaler()
X_emotion_train_scaled = emotion_scaler.fit_transform(X_emotion_train)

emotion_model = RandomForestClassifier(n_estimators=100, random_state=42)
emotion_model.fit(X_emotion_train_scaled, y_emotion_train)

with open('model/emotion_model.pkl', 'wb') as emotion_model_file:
    pickle.dump(emotion_model, emotion_model_file)

with open('model/emotion_scaler.pkl', 'wb') as emotion_scaler_file:
    pickle.dump(emotion_scaler, emotion_scaler_file)

print("\nModel and Scaler Saved Successfully!")