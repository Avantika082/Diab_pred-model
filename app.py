from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load your trained machine learning model
model = pickle.load(open('diab_pred.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extracting input data from the form
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # Predicting using the loaded model
    prediction = model.predict(final_features)
    
    # Check result and return appropriate message
    output = prediction[0]
    
    if output == 1:
        result = 'You are likely to have diabetes.'
    else:
        result = 'You are unlikely to have diabetes.'
    
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
