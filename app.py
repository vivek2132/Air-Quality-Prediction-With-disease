from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained Random Forest model and label mapping
model = pickle.load(open('model.pkl', 'rb'))

label_mapping = {'High': 0, 'Low': 1, 'Moderate': 2, 'Very High': 3, 'Very low': 4}
inverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Pollution level details
pollution_levels = {
    "Very Low": {
        "diseases": ["No reported health risks"],
        "preventions": ["Avoid activities that release pollutants, such as open burning or excessive vehicle use","Support policies promoting renewable energy and reducing industrial emissions"]
    },
    "Low": {
        "diseases": ["Generally safe, though hypersensitive individuals may experience mild symptoms"],
        "preventions": ["Avoid burning waste or using polluting fuels indoors", "Use air quality monitors to ensure continued good quality"]
    },
    "Moderate": {
        "diseases": ["Mild respiratory irritation (coughing, throat discomfort)", "Mild cardiovascular stress in sensitive individuals"],
        "preventions": ["Limit time spent near traffic or industrial areas", "Drink plenty of water to flush toxins","Keep windows open during periods of cleaner air","Use saline sprays or humidifiers to reduce irritation"]
    },
    "High": {
        "diseases": ["Persistent asthma symptoms, respiratory infections, reduced lung function", "Increased hypertension and angina risk","Eye irritation and sinus issues"],
        "preventions": ["Use air purifiers with activated carbon filters", "Wear masks during high-pollution periods","Stay indoors during peak pollution hours, typically morning and evening","Ensure windows and doors are sealed to prevent pollution from entering","Use indoor plants that help purify air, such as spider plants or peace lilies"]
    },
    "Very High": {
        "diseases": ["Severe asthma attacks, COPD exacerbations, acute bronchitis", "Heart attacks, arrhythmias, hypertension","Increased risk of strokes, cognitive decline, cancer","Premature death in high-risk groups"],
        "preventions": ["Limit outdoor activities and stay in well-ventilated spaces", " Use high-quality HEPA air purifiers indoors","Wear N95 or higher-grade masks when outdoors","Refrain from exercising outdoors"]
    }
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data
        data = [float(request.form[feature]) for feature in ['CO_GT', 'PT08_S1_CO', 'C6H6_GT', 
                                                             'PT08_S2_NMHC', 'Nox_GT', 'PT08_S3_Nox', 
                                                             'NO2_GT', 'PT08_S4_NO2', 'PT08_S5_O3', 
                                                             'T', 'RH', 'AH']]
        
        # Create DataFrame for prediction
        features = pd.DataFrame([data], columns=['CO_GT', 'PT08_S1_CO', 'C6H6_GT', 'PT08_S2_NMHC', 
                                                 'Nox_GT', 'PT08_S3_Nox', 'NO2_GT', 'PT08_S4_NO2', 
                                                 'PT08_S5_O3', 'T', 'RH', 'AH'])

        # Predict air quality level
        prediction = model.predict(features)[0]
        predicted_label = inverse_label_mapping[prediction]

        # Fetch diseases and prevention info
        level_data = pollution_levels.get(predicted_label, {"diseases": [], "preventions": []})

        return render_template('index.html', 
                               prediction_text=f'Predicted Pollution Level: {predicted_label}',
                               diseases=level_data["diseases"], 
                               preventions=level_data["preventions"])

    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', prediction_text="An error occurred during prediction.")

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
