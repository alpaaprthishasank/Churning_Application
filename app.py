from flask import Flask, render_template, request
import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

app = Flask(__name__)

# Load models and scalers
ss1 = joblib.load('standard_scaler_llm.pkl')
dt_clf1 = joblib.load('decision_tree_model_llm.pkl')

lr_clfs = {}
for segment in dt_clf1.classes_:
    lr_clfs[segment] = joblib.load(f'logistic_regression_model_segment_{segment}.pkl')

ss = joblib.load('standard_scaler_svm.pkl')
dt_clf = joblib.load('decision_tree_model_svm.pkl')

svm_clfs = {}
for segment in dt_clf.classes_:
    svm_clfs[segment] = joblib.load(f'svm_model_segment_{segment}_svm.pkl')

rf_clf = joblib.load("rf_clf_resampled_rlm.pkl")

lr_clfs_rf_resampled_loaded = {}
for segment in rf_clf.classes_:
    lr_clfs_rf_resampled_loaded[segment] = joblib.load(f'logistic_regression_model_segment_{segment}_rlm.pkl')

ss = joblib.load("standard_scaler_rlm.pkl")

#ensemble
loaded_raw = []
for idx in range(5):
    with open(f'model_raw_{idx}.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
        loaded_raw.append(loaded_model)

with open("standard_scaler_raw.pkl", "rb") as f:
    ss_raw= pickle.load(f)
    
education_level_mapping = {
    'Unknown': 0,
    'Uneducated': 1,
    'High School': 2,
    'College': 3,
    'Graduate': 4,
    'Post-Graduate': 5,
    'Doctorate': 6
}

marital_status_mapping = {
    'Unknown': 0,
    'Single': 1,
    'Married': 2,
    'Divorced': 3
}

income_category_mapping = {
    'Unknown': 0,
    'Less than $40K': 1,
    '$40K - $60K': 2,
    '$60K - $80K': 3,
    '$80K - $120K': 4,
    '$120K +': 5
}

card_category_mapping = {
    'Blue': 0,
    'Silver': 1,
    'Gold': 2,
    'Platinum': 3
}
##home
@app.route('/')
def home():
   return render_template('home.html')

@app.route('/index2')
def index2():
    return render_template('index2.html')

#@app.route('/',)
 
@app.route('/index1')
def index1():
    return render_template('index1.html')

@app.route('/tele')
def tele():
    return render_template('tele.html')
# load pickle files
##bankchurn models
#rlm
# Load RandomForestClassifier model
with open("rf_clf_bank.pkl", "rb") as f:
    rf_clf_bank = pickle.load(f)

lr_models_bank = {}
for segment in rf_clf_bank.classes_:
    with open(f"lr_model_bank_segment_{segment}.pkl", "rb") as f:
        lr_model = pickle.load(f)
        lr_models_bank[segment] = lr_model

# Load StandardScaler object
with open("standard_bank_scaler.pkl", "rb") as f:
    ss_bank = pickle.load(f)


#llm
with open("dt_clf_bank_llm.pkl", "rb") as f:
    dt_clf_bank_llm = pickle.load(f)

lr_models_bank_llm = {}
for segment in rf_clf_bank.classes_:
    with open(f"lr_model_bank_segment_llm_{segment}.pkl", "rb") as f:
        lr_model = pickle.load(f)
        lr_models_bank_llm[segment] = lr_model

# Load StandardScaler object
with open("standard_bank_scaler_llm.pkl", "rb") as f:
    ss_bank_llm = pickle.load(f)



#slm
# Load StandardScaler object
with open("standard_bank_scaler_slm.pkl", "rb") as f:
    ss_bank_slm = pickle.load(f)

with open("dt_clf_bank_slm.pkl", "rb") as f:
    dt_clf_bank_slm = pickle.load(f)

svm_clfs_bank = {}
for segment in dt_clf_bank_slm.classes_:
    with open(f"svm_model_bank_segment_slm_{segment}.pkl", "rb") as f:
        svm_model = pickle.load(f)
        svm_clfs_bank[segment] =svm_model

#ensemble
loaded_bank = []
for idx in range(3):
    with open(f'model_bank_{idx}.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
        loaded_bank.append(loaded_model)

with open("standard_scaler_bank.pkl", "rb") as f:
    ss_bank= pickle.load(f)
    

#telecom 
#pickle files
#rlm
with open("rf_clf_tele_rlm.pkl", "rb") as f:
    rf_clf_tel_rlm = pickle.load(f)

lr_models_tel_rlm = {}
for segment in rf_clf_tel_rlm.classes_:
    with open(f"lr_model_tele_segment_rlm_{segment}.pkl", "rb") as f:
        lr_model = pickle.load(f)
        lr_models_tel_rlm[segment] = lr_model

# Load StandardScaler object
with open("standard_tele_scaler_rlm.pkl", "rb") as f:
    ss_tel_rlm = pickle.load(f)

#llm
with open("dt_clf_tele_llm.pkl", "rb") as f:
    dt_clf_tel_llm = pickle.load(f)

lr_models_tel_llm = {}
for segment in dt_clf_tel_llm.classes_:
    with open(f"lr_model_tele_segment_rlm_{segment}.pkl", "rb") as f:
        lr_model = pickle.load(f)
        lr_models_tel_llm[segment] = lr_model

# Load StandardScaler object
with open("standard_tele_scaler_llm.pkl", "rb") as f:
    ss_tel_llm = pickle.load(f)

#slm
with open("standard_tele_scaler_slm.pkl", "rb") as f:
    ss_tele_slm = pickle.load(f)

with open("dt_clf_tele_slm.pkl", "rb") as f:
    dt_clf_tele_slm = pickle.load(f)

svm_clfs_tele_slm = {}
for segment in dt_clf_bank_slm.classes_:
    with open(f"svm_model_tele_segment_slm_{segment}.pkl", "rb") as f:
        svm_model = pickle.load(f)
        svm_clfs_tele_slm[segment] =svm_model
    
#ensemble
loaded_tele = []
for idx in range(3):
    with open(f'model_tele_{idx}.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
        loaded_tele.append(loaded_model)

with open("standard_scaler_tele.pkl", "rb") as f:
    ss_tele= pickle.load(f)
    

@app.route('/submit_prediction', methods=['POST'])
def submit_prediction():
    if request.method == 'POST':
        count = request.form['count']
        country = request.form['country']
        state = request.form['state']
        city = request.form['city']
        zip_code = request.form['zip_code']
        gender = request.form['gender']
        senior_citizen = request.form['senior_citizen']
        partner = request.form['partner']
        dependents = request.form['dependents']
        tenure = request.form['tenure']
        phone_service = request.form['phone_service']
        multiple_lines = request.form['multiple_lines']
        online_security = request.form['online_security']
        online_backup = request.form['online_backup']
        device_protection = request.form['device_protection']
        tech_support = request.form['tech_support']
        streaming_tv = request.form['streaming_tv']
        streaming_movies = request.form['streaming_movies']
        paperless_billing = request.form['paperless_billing']
        monthly_charges = request.form['monthly_charges']
        total_charges = request.form['total_charges']
        gender_encoded = 1 if gender == 'M' else 0
        country_encode=1 if country=='United States' else 0
        state_encode=1 if state=='California' else 0
        # Now you have all the form data, you can process it as needed
        # For example, you can pass it to a machine learning model for prediction

        # For now, let's just print the data to the console
        #count,country,state,city,zip_code,
        new_data_tele = [[count,country_encode,state_encode,zip_code,gender_encoded,senior_citizen,partner,
                     dependents,tenure,phone_service,multiple_lines,online_security,online_backup,device_protection
                     ,tech_support,streaming_tv,streaming_movies,paperless_billing,monthly_charges,
                     total_charges]]
      

        # Create a DataFrame from the new data
        df_new_data_tel_rlm = pd.DataFrame(new_data_tele)
        #RLM
        # Scale the features
        X_new_data_scaled_tel_rlm = ss_tel_rlm.transform(df_new_data_tel_rlm)

        # Predict using the trained models
        predictions_new_data_tel_rlm = []

        for x in X_new_data_scaled_tel_rlm:
            segment_rf_resampled_tel_rlm = rf_clf_tel_rlm.predict([x])[0]
            if isinstance(lr_models_tel_rlm[segment_rf_resampled_tel_rlm], LogisticRegression):
                lr_model_rf_resampled_tel_rlm = lr_models_tel_rlm[segment_rf_resampled_tel_rlm]
                y_pred = lr_model_rf_resampled_tel_rlm.predict([x])[0]
                predictions_new_data_tel_rlm.append(y_pred)
            else:
                predictions_new_data_tel_rlm.append(lr_models_bank[segment_rf_resampled_tel_rlm])

        print(predictions_new_data_tel_rlm[0])
        #llm
        df_new_data_tel_llm = pd.DataFrame(new_data_tele)
        
        # Scale the features
        X_new_data_scaled_tel_llm = ss_tel_llm.transform(df_new_data_tel_llm)

        # Predict using the trained models
        predictions_new_data_tel_llm = []

        for x in X_new_data_scaled_tel_llm:
            segment_rf_resampled_tel_llm = dt_clf_tel_llm.predict([x])[0]
            if isinstance(lr_models_tel_llm[segment_rf_resampled_tel_llm], LogisticRegression):
                lr_model_rf_resampled_tel_llm = lr_models_tel_llm[segment_rf_resampled_tel_llm]
                y_pred = lr_model_rf_resampled_tel_llm.predict([x])[0]
                predictions_new_data_tel_llm.append(y_pred)
            else:
                predictions_new_data_tel_llm.append(lr_models_tel_llm[segment_rf_resampled_tel_llm])
        
        print(predictions_new_data_tel_llm[0])
        #slm
        df_new_data_tel_slm = pd.DataFrame(new_data_tele)
        
        # Scale the features
        X_new_data_scaled_tel_slm = ss_tel_llm.transform(df_new_data_tel_slm)

        # Predict using the trained models
        predictions_new_data_tel_slm = []

        for x in X_new_data_scaled_tel_slm:
            segment_rf_resampled_tel_slm = dt_clf_tele_slm.predict([x])[0]
            if isinstance(svm_clfs_tele_slm[segment_rf_resampled_tel_slm], LogisticRegression):
                svm_model_rf_resampled_tel_slm = svm_clfs_tele_slm[segment_rf_resampled_tel_llm]
                y_pred = svm_model_rf_resampled_tel_slm.predict([x])[0]
                predictions_new_data_tel_slm.append(y_pred)
            else:
                predictions_new_data_tel_slm.append(svm_clfs_tele_slm[segment_rf_resampled_tel_llm])
        
        print(predictions_new_data_tel_slm[0])
        #ensemble
        # Scale the features
        df_new_data_tel_ensemble= pd.DataFrame(new_data_tele)
        X_new_data_scaled = ss_tele.transform(df_new_data_tel_ensemble)

         # Predict using the trained ensemble model
        y_preds_ensemble_new_data = np.mean([model.predict(X_new_data_scaled) for model in loaded_tele], axis=0)
        predictions_new_data_ensemble = (y_preds_ensemble_new_data > 0.5).astype(int)


# Display predictions for the new data
        print("Predictions for the new data:")
        print(predictions_new_data_ensemble[0])
        return render_template("tele.html",predict=predictions_new_data_tel_rlm[0],predict1=predictions_new_data_tel_llm[0],predict2=predictions_new_data_tel_slm[0],
                               predict3=predictions_new_data_ensemble[0])
    


# bank churn dataset
@app.route('/predict1', methods=['POST'])
def predict1():
    if request.method == 'POST':
        customer_age = float(request.form['Customer_Age'])
        dependent_count = float(request.form['Dependent_count'])
        months_on_book = float(request.form['Months_on_book'])
        total_relationship_count = float(request.form['Total_Relationship_Count'])
        months_inactive_12_mon = float(request.form['Months_Inactive_12_mon'])
        contacts_count_12_mon = float(request.form['Contacts_Count_12_mon'])
        credit_limit = float(request.form['Credit_Limit'])
        total_revolving_bal = float(request.form['Total_Revolving_Bal'])
        avg_open_to_buy = float(request.form['Avg_Open_To_Buy'])
        total_amt_chng_q4_q1 = float(request.form['Total_Amt_Chng_Q4_Q1'])
        total_trans_amt = float(request.form['Total_Trans_Amt'])
        total_trans_ct = float(request.form['Total_Trans_Ct'])
        total_ct_chng_q4_q1 = float(request.form['Total_Ct_Chng_Q4_Q1'])
        avg_utilization_ratio = float(request.form['Avg_Utilization_Ratio'])
        gender = request.form['Gender']
        education_level = request.form['Education_Level']
        marital_status = request.form['Marital_Status']
        income_category = request.form['Income_Category']
        card_category = request.form['Card_Category']

        # Process categorical variables
        gender_encoded = 1 if gender == 'M' else 0
        education_level_encoded = education_level_mapping[education_level]
        marital_status_encoded = marital_status_mapping[marital_status]
        income_category_encoded = income_category_mapping[income_category]
        card_category_encoded = card_category_mapping[card_category]

        # Create a new data point
        new_data = [[customer_age, dependent_count, months_on_book, total_relationship_count,
                     months_inactive_12_mon, contacts_count_12_mon, credit_limit, total_revolving_bal,
                     avg_open_to_buy, total_amt_chng_q4_q1, total_trans_amt, total_trans_ct,
                     total_ct_chng_q4_q1, avg_utilization_ratio, gender_encoded,
                     education_level_encoded, marital_status_encoded, income_category_encoded,
                     card_category_encoded]]

        # Create a DataFrame from the new data
        df_new_data = pd.DataFrame(new_data)
        #RLM
        # Scale the features
        X_new_data_scaled = ss_bank.transform(df_new_data)

        # Predict using the trained models
        predictions_new_data = []

        for x in X_new_data_scaled:
            segment_rf_resampled = rf_clf_bank.predict([x])[0]
            if isinstance(lr_models_bank[segment_rf_resampled], LogisticRegression):
                lr_model_rf_resampled = lr_models_bank[segment_rf_resampled]
                y_pred = lr_model_rf_resampled.predict([x])[0]
                predictions_new_data.append(y_pred)
            else:
                predictions_new_data.append(lr_models_bank[segment_rf_resampled])
        print(predictions_new_data)
        print(predictions_new_data)

        #LLM
        # Scale the features
        X_new_data_scaled_llm = ss_bank_llm.transform(df_new_data)

        # Predict using the trained models
        predictions_new_data_llm = []

        for x in X_new_data_scaled_llm:
            segment_rf_resampled_llm = dt_clf_bank_llm.predict([x])[0]
            if isinstance(lr_models_bank_llm[segment_rf_resampled_llm], LogisticRegression):
                lr_model_rf_resampled_llm = lr_models_bank_llm[segment_rf_resampled]
                y_pred = lr_model_rf_resampled_llm.predict([x])[0]
                predictions_new_data_llm.append(y_pred)
            else:
                predictions_new_data_llm.append(lr_models_bank_llm[segment_rf_resampled_llm])
        print(predictions_new_data_llm)
        print(predictions_new_data_llm)


         #SLM
        # Scale the features
        X_new_data_scaled_slm = ss_bank_llm.transform(df_new_data)

        # Predict using the trained models
        predictions_new_data_slm = []

        for x in X_new_data_scaled_slm:
            segment_rf_resampled_slm = dt_clf_bank_slm.predict([x])[0]
            if isinstance(svm_clfs_bank[segment_rf_resampled_slm], LogisticRegression):
                svc_model_rf_resampled_llm = svm_clfs_bank[segment_rf_resampled_slm]
                y_pred =svc_model_rf_resampled_llm.predict([x])[0]
                predictions_new_data_slm.append(y_pred)
            else:
                predictions_new_data_slm.append(svm_clfs_bank[segment_rf_resampled_slm])
        print(predictions_new_data_slm)
        print(predictions_new_data_slm)

         #ensemble
        # Scale the features
        df_new_data_bank_ensemble= pd.DataFrame(new_data)
        X_new_data_scaled_bank = ss_bank.transform(df_new_data_bank_ensemble)

         # Predict using the trained ensemble model
        ensemble_new_data = np.mean([model.predict(X_new_data_scaled_bank) for model in loaded_bank], axis=0)
        predictions_new_data_ensemble_bank = (ensemble_new_data > 0.5).astype(int)
        print("ensemble",predictions_new_data_ensemble_bank[0])

# Display predictions for the new data
        print("Predictions for the new data:")
        return render_template('index1.html', predict=predictions_new_data[0],predict1=predictions_new_data_llm[0],predict2=predictions_new_data_slm[0],predict3=predictions_new_data_ensemble_bank[0])


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
        
         #ensemble
        # Scale the features
        df_new_data_raw_ensemble= pd.DataFrame(new_data, columns=features)
        X_new_data_scaled_raw = ss_raw.transform(df_new_data_raw_ensemble)

         # Predict using the trained ensemble model
        ensemble_new_data_raw = np.mean([model.predict(X_new_data_scaled_raw) for model in loaded_raw], axis=0)
        predictions_new_data_ensemble_raw = (ensemble_new_data_raw > 0.5).astype(int)
        print("ensemble",predictions_new_data_ensemble_raw[0])

        return render_template('index2.html', predict=prediction, predict1=prediction1, predict2=prediction2,predict3=predictions_new_data_ensemble_raw[0])


if __name__ == '__main__':
    app.run()
