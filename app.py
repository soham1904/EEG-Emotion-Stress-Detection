from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

try:
    with open('model/stress_model.pkl', 'rb') as stress_model_file, open('model/stress_scaler.pkl',
                                                                         'rb') as stress_scaler_file:
        stress_model = pickle.load(stress_model_file)
        stress_scaler = pickle.load(stress_scaler_file)

    with open('model/emotion_model.pkl', 'rb') as emotion_model_file, open('model/emotion_scaler.pkl',
                                                                           'rb') as emotion_scaler_file:
        emotion_model = pickle.load(emotion_model_file)
        emotion_scaler = pickle.load(emotion_scaler_file)
except FileNotFoundError:
    raise Exception("Models or scalers not found. Train and save them first.")


@app.route('/')
def home():

    """Renders the main HTML page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests."""
    try:

        data = request.json.get('data')
        model_type = request.json.get('model_type')  # 'stress' or 'emotion'

        if not data or not model_type:
            return jsonify({'error': 'All fields are required.'}), 400

        features = np.array(data, dtype=float).reshape(1, -1)
        print(f"Input features: {features}")

        if model_type == 'stress':
            if features.shape[1] != 4:
                return jsonify({'error': 'Stress prediction requires exactly 4 inputs.'}), 400

            scaled_features = stress_scaler.transform(features)
            print(f"Scaled stress features: {scaled_features}")

            prediction = stress_model.predict(scaled_features)[0]
            print(f"Stress prediction result: {prediction}")

            result = prediction

        elif model_type == 'emotion':
            if features.shape[1] != 1:
                return jsonify({'error': 'Emotion prediction requires exactly 1 input.'}), 400

            scaled_features = emotion_scaler.transform(features)
            print(f"Scaled emotion features: {scaled_features}")

            prediction = emotion_model.predict(scaled_features)[0]
            print(f"Emotion prediction result: {prediction}")

            result = f"Emotion: {prediction}"

        else:
            return jsonify({'error': 'Invalid model type.'}), 400

        return jsonify({'prediction': result})

    except ValueError as e:
        return jsonify({'error': f'Invalid input data: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
