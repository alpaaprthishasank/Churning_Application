from flask import Flask, render_template, request
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  # Add this line to import SVC
import os
app = Flask(__name__)

#LLM
ss1 = joblib.load('standard_scaler_llm.pkl')
dt_clf1 = joblib.load('decision_tree_model_llm.pkl')

lr_clfs= {}
for segment in dt_clf1.classes_:
    lr_clfs[segment] = joblib.load(f'logistic_regression_model_segment_{segment}.pkl')
#svm
ss = joblib.load('standard_scaler_svm.pkl')
dt_clf = joblib.load('decision_tree_model_svm.pkl')

svm_clfs= {}
for segment in dt_clf.classes_:
    svm_clfs[segment] = joblib.load(f'svm_model_segment_{segment}_svm.pkl')

#rlm
rf_clf = joblib.load("rf_clf_resampled_rlm.pkl")

# Load LogisticRegression models
lr_clfs_rf_resampled_loaded = {}
for segment in rf_clf.classes_:
    lr_clfs_rf_resampled_loaded [segment] = joblib.load(f'logistic_regression_model_segment_{segment}_rlm.pkl')

# Load StandardScaler object
ss=joblib.load("standard_scaler_rlm.pkl")
    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index1')
def index1():
    return render_template('index1.html')
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract form data
        creditScore = float(request.form['CreditScore'])
        age = float(request.form['Age'])
        tenure = float(request.form['Tenure'])
        balance = float(request.form['Balance'])
        numOfProducts = float(request.form['NumOfProducts'])
        hasCrCard = int(request.form['HasCrCard'])
        isActiveMember = int(request.form['IsActiveMember'])
        estimatedSalary = float(request.form['EstimatedSalary'])
        zeroBalance = int(request.form['ZeroBalance'])
        female = int(request.form['Female'])
        male = int(request.form['Male'])
        france = int(request.form['France'])
        germany = int(request.form['Germany'])
        spain = int(request.form['Spain'])

        # Prepare data for prediction
        new_data = np.array([creditScore, age, tenure, balance, numOfProducts, hasCrCard, isActiveMember,
                             estimatedSalary, zeroBalance, female, male, france, germany, spain]).reshape(1, -1)

        # Convert the given data to DataFrame
        features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                    'EstimatedSalary', 'ZeroBalance', 'Female', 'Male', 'France', 'Germany', 'Spain']
        df_new_data = pd.DataFrame(new_data, columns=features)

        # Scale the features
        X_new_data_scaled = ss.transform(df_new_data)

        # Predict using the trained model
        predictions_new_data = []
        for x in X_new_data_scaled:
            segment = dt_clf.predict([x])[0]
            if isinstance(svm_clfs[segment], SVC):
                svm_clf = svm_clfs[segment]
                y_pred = svm_clf.predict([x])[0]
                predictions_new_data.append(y_pred)
            else:
                predictions_new_data.append(svm_clfs[segment])

        prediction = predictions_new_data[0]  # Assuming only one prediction is made
    #llm
        df_new_data1 = pd.DataFrame(new_data, columns=features)

# Scale the features
        X_new_data_scaled = ss1.transform(df_new_data1)

# Predict using the trained model
        predictions_new_data_llm = []

        for x in X_new_data_scaled:
           
           segment = dt_clf1.predict([x])[0]
           lr_clf = lr_clfs[segment]
           if isinstance(lr_clf, LogisticRegression):  # Ensure lr_clf is a model object
              y_pred = lr_clf.predict([x])[0]
              predictions_new_data_llm.append(y_pred)
           else:
            print(f"Segment {segment} contains only one class. Assigning majority class as the predicted class.")
            majority_class = lr_clf  # Majority class of the segment
            predictions_new_data_llm.append(majority_class)
            prediction1 = predictions_new_data_llm[0]
#RLM
            df_new_data = pd.DataFrame(new_data, columns=features)

# Scale the features
            X_new_data_scaled = ss.transform(df_new_data)

# Predict using the trained model
# Predict using the trained model
            predictions_new_data_rlm = []

           for x in X_new_data_scaled:
            segment_rf_resampled = rf_clf.predict([x])[0]  # Use rf_clf instead of rf_clf_resampled
            if isinstance(lr_clfs_rf_resampled_loaded[segment_rf_resampled], LogisticRegression):
             lr_clf_rf_resampled = lr_clfs_rf_resampled_loaded[segment_rf_resampled]
             y_pred = lr_clf_rf_resampled.predict([x])[0]
             predictions_new_data_rlm.append(y_pred)
            else:
              predictions_new_data_rlm.append(lr_clfs_rf_resampled_loaded[segment_rf_resampled])

# Display predictions for the given data

        prediction2 = predictions_new_data_rlm[0]
        return render_template('index.html', predict=prediction,predict1=prediction1,predict2=prediction2)

if __name__ == '__main__':
    app.run()


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract form data
        creditScore = float(request.form['CreditScore'])
        age = float(request.form['Age'])
        tenure = float(request.form['Tenure'])
        balance = float(request.form['Balance'])
        numOfProducts = float(request.form['NumOfProducts'])
        hasCrCard = int(request.form['HasCrCard'])
        isActiveMember = int(request.form['IsActiveMember'])
        estimatedSalary = float(request.form['EstimatedSalary'])
        zeroBalance = int(request.form['ZeroBalance'])
        gender = request.form['Gender']
        country = request.form['Country']

        # Map 'Country' to binary variables
        france = 1 if country == 'France' else 0
        germany = 1 if country == 'Germany' else 0
        spain = 1 if country == 'Spain' else 0

        # Map 'Gender' to binary variable
        if gender == 'Male':
            male = 1
            female = 0
        else:
            male = 0
            female = 1

        # Prepare data for prediction
        new_data = np.array([creditScore, age, tenure, balance, numOfProducts, hasCrCard, isActiveMember,
                             estimatedSalary, zeroBalance, female, male, france, germany, spain]).reshape(1, -1)

        # Convert data to DataFrame
        features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                    'EstimatedSalary', 'ZeroBalance', 'Female', 'Male', 'France', 'Germany', 'Spain']
        df_new_data = pd.DataFrame(new_data, columns=features)

        # Scale the features
        X_new_data_scaled = ss.transform(df_new_data)

        # Predict using the trained models
        segment = dt_clf.predict(X_new_data_scaled)[0]

        if isinstance(svm_clfs[segment], SVC):
            svm_clf = svm_clfs[segment]
            prediction = svm_clf.predict(X_new_data_scaled)[0]
        else:
            prediction = svm_clfs[segment]

        segment_llm = dt_clf1.predict(X_new_data_scaled)[0]
        lr_clf = lr_clfs[segment_llm]
        if isinstance(lr_clf, LogisticRegression):
            prediction1 = lr_clf.predict(X_new_data_scaled)[0]
        else:
            majority_class = lr_clf
            prediction1 = majority_class

        segment_rlm = rf_clf.predict(X_new_data_scaled)[0]
        lr_clf_rf_resampled = lr_clfs_rf_resampled_loaded[segment_rlm]
        if isinstance(lr_clf_rf_resampled, LogisticRegression):
            prediction2 = lr_clf_rf_resampled.predict(X_new_data_scaled)[0]
        else:
            prediction2 = lr_clf_rf_resampled

        return render_template('index2.html', predict=prediction, predict1=prediction1, predict2=prediction2)
