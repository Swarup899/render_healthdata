from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open('diabetes_heartdisease_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    data = pickle.load(f)
    label_encoders = data['label_encoders']
    disease_le = data['disease_le']

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    gender = label_encoders['Gender'].transform([request.form['gender']])[0]
    bmi = float(request.form['bmi'])
    bp = int(request.form['bp'])
    sugar = int(request.form['sugar'])
    cholesterol = int(request.form['cholesterol'])
    smoking = label_encoders['Smoking'].transform([request.form['smoking']])[0]
    family_history = label_encoders['FamilyHistory'].transform([request.form['family_history']])[0]

    features = np.array([[age, gender, bmi, bp, sugar, cholesterol, smoking, family_history]])

    prediction = model.predict(features)[0]
    disease = disease_le.inverse_transform([prediction])[0]

    return jsonify({"result": f"You Have: {disease}"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')  