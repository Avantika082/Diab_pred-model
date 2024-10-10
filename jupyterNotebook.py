# DIABETES PREDICTION MODEL 

# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# Load the dataset
data = pd.read_csv(r"diabetes.csv")


# REMOVING OUTLIERS

data1 = data[(data['Pregnancies']<13) & (data['Glucose']<180) & (data['Glucose']>49) & (data['BloodPressure']<102) & (data['BloodPressure']>42) & (data['Insulin']<310) & (data['BMI']>21) & (data['BMI']<49) & (data['DiabetesPedigreeFunction']<1.05) & (data['Age']<62)]


data1.head()

# STANDARD SCALING AND LABEL ENCODING

x = data1.drop(['Outcome'] , axis = 1)
y = data1['Outcome']


# Splitting the data into train and test data

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3 , random_state = 0 , stratify = y) 


# Scaling the data

sc = StandardScaler()
sc.fit_transform(x_train)
sc.transform(x_test)


# FINDING THE BEST MODEL THROUGH HYPERPARAMETER TUNING

# Import neccessary models

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import GridSearchCV


# Creating list of models with parameters for hyperparameter tuning
model_params = {
    'svm' : {
        'model' : svm.SVC(gamma = 'auto'),
        'params' : {
            'C' : [1,10,20] ,
            'kernel' : ['rbf' , 'linear']
        }

    },
    'LogisticRegression' : {
        'model' : LogisticRegression(solver = 'liblinear' , multi_class = 'auto' ),
        'params' : {
            'C' : [1,20,50]
        }
    },
    'Kneighbors_classifier' : {
        'model' : KNeighborsClassifier(),
        'params' : {
           'n_neighbors' : [1,10,15,20,25,30]
        }
    }
}            


# Hyperparameter tuning
scores =[]

for model_name,mp in model_params.items():
    clf = GridSearchCV(mp['model'] , mp['params'] , cv=5 , return_train_score = False)
    clf.fit(x_train,y_train)
    scores.append({
        'model' : model_name ,
        'best_score' : clf.best_score_ ,
        'best_params' : clf.best_params_
    })
  


# Creating a dataframe for best model with best parameters
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df

# Fit the model to selected machine learning model
model = svm.SVC(gamma = 'auto' , C = 1 , kernel = 'linear')
model.fit(x_train,y_train)


# Predict the test values
y_pred = model.predict(x_test)


# ACCURACY OF THE MODEL

# Confusion metrics code
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test , y_pred)


# Classification report code
from sklearn.metrics import classification_report
print(classification_report(y_test , y_pred))

#Converting the model into pickle file
import pickle
with open('diab_pred.pkl' , 'wb') as f:
    pickle.dump(model,f)

