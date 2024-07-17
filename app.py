from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the machine learning model
model = joblib.load(r"model\Alzheimer's Disease Model.h5")

# Define a dictionary to map numerical predictions to labels
prediction_labels = {0: "Nondemented", 1: "Demented"}

# Function to preprocess input data
def preprocess_input(M_F, Age, EDUC, SES, MMSE, CDR, eTIV, nWBV, ASF):
    # Example preprocessing steps:
    # You should adapt this based on how your model expects the input
    input_data = np.array([[M_F, Age, EDUC, SES, MMSE, CDR, eTIV, nWBV, ASF]])
    return input_data

# Route to render the prediction form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission and make predictions
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Retrieve form data
        M_F = int(request.form['M/F'])
        Age = float(request.form['Age'])
        EDUC = float(request.form['EDUC'])
        SES = float(request.form['SES'])
        MMSE = float(request.form['MMSE'])
        CDR = float(request.form['CDR'])
        eTIV = float(request.form['eTIV'])
        nWBV = float(request.form['nWBV'])
        ASF = float(request.form['ASF'])
        
        # Preprocess input data
        input_data = preprocess_input(M_F, Age, EDUC, SES, MMSE, CDR, eTIV, nWBV, ASF)
        
        # Make predictions
        numerical_prediction = model.predict(input_data)[0]  # Assuming a single prediction
        label_prediction = prediction_labels[numerical_prediction]  # Map numerical prediction to label
        
        # You might need additional post-processing of the prediction
        
        # Return the prediction label to the prediction page along with the return button
        return render_template('predict.html', result=label_prediction)
    else:
        # If GET request, return Method Not Allowed error
        return "Method Not Allowed", 405

if __name__ == '__main__':
    app.run(debug=True)








   
