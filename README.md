This project is a machine learning model that predicts the likelihood of an individual having diabetes based on specific health metrics such as glucose level, blood pressure, BMI, and others. The project is built using Python, trained on the Pima Indians Diabetes dataset, and deployed using Flask as a web application.


### **Table of Contents**

- Project Overview
- Technologies Used
- Features
- Usage
- Model Explanation
- Project Structure
- Contributing




### **Project Overview**


The Diabetes Prediction Model is a classification model that predicts whether a person has diabetes based on several input parameters such as:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age


The web application takes input from the user and provides the prediction through a trained machine learning model.




### **Technologies Used**


- Python: Programming language for model development.
- Flask: Web framework for deploying the model as a web app.
- Pandas & Numpy: For data manipulation and processing.
- scikit-learn: For building the machine learning model.
- HTML/CSS: For front-end user interface design.




### **Features**


User-friendly Web Interface: Input health data and get predictions through a simple web interface.
Machine Learning Model: Hypertuned svm trained on the Diabetes dataset.
Prediction: Provides a binary prediction (positive or negative for diabetes).




### **Setup Instructions**


Prerequisites:

Python 3.x installed.


Virtual environment (recommended).


Required Python libraries listed in requirements.txt.




### **Usage**


Once the Flask application is running:

Enter the required health metrics such as glucose levels, BMI, etc. on the provided web form.
Submit the form to receive a prediction on whether the individual is likely to have diabetes or not.




### **Model Explanation:**

The model is based on a svm and was trained on the Pima Indians Diabetes Dataset. The model uses various health-related features to predict the likelihood of diabetes. The dataset consists of 768 samples and 8 features, including pregnancies ,skin thickness , BMI ,  glucose level, insulin levels, and age.




### Training the Model:

The dataset is preprocessed (handling missing values, scaling features).
The data is split into training and testing sets.
A List model is trained including support vector machine(svm) , logistic regression and k-neighbors classifier are hypertuned and best model with best parameters is trained, evaluated, and saved as a .pkl file to be used for predictions in the web app.




### **Project Structure**

diabetes-prediction-model/

- ├── app.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                 # Flask application
- ├── jupyterNotebook.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     # Machine learning model logic
- ├── diab_pred.pkl&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;          # Saved trained model for predictions
- ├── requirements.txt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       # Python dependencies
- ├── templates/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;             # HTML files for the web interface
- │   └── index.html&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;         # Main page for user input
- ├── static/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                # CSS and other static assets
- │   └── style.css&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;          # Stylesheet for web page
- └── README.md&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;              # Project documentation (this file)




### **Contributing**


Feel free to open issues or pull requests if you have suggestions or improvements. Please ensure your contributions adhere to the project's coding guidelines.