 
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier


# loading the trained model
model_path = "./model.pkl"
scaler_path = "./scaler.pkl"
model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

def gender_transformer(df, gender):
  if gender == 'Male':
    df['gender_Male'] = [1]
    df['gender_Other'] = [0]
  elif gender == 'Other':
    df['gender_Other'] = [1]
    df['gender_Male'] = [0]
  else:
    df['gender_Male'] = [0]
    df['gender_Other'] = [1]


def hypertension_transformer(df, hypertension):
  if hypertension == 'Yes':
    df['hypertension'] = [1]
  else:
    df['hypertension'] = [0]

def heart_disease_transformer(df, heart_disease):
  if heart_disease == 'Yes':
    df['heart_disease'] = [1]
  else:
    df['heart_disease'] = [0]

def marriage_transformer(df, married):
  if married == 'Yes':
    df['ever_married_Yes'] = [1]
  else:
    df['ever_married_Yes'] = [0]

def worktype_transformer(df, work_type):
  if work_type == 'Never_worked':
    df['work_type_Never_worked'] = [1]
    df['work_type_Private'] = [0]
    df['work_type_children'] = [0]
    df['work_type_Self-employed'] = [0]
  elif work_type == 'Private':
    df['work_type_Never_worked'] = [0]
    df['work_type_Private'] = [1]
    df['work_type_children'] = [0]
    df['work_type_Self-employed'] = [0]
  elif work_type == 'Self-employed':
    df['work_type_Never_worked'] = [0]
    df['work_type_Private'] = [0]
    df['work_type_children'] = [0]
    df['work_type_Self-employed'] = [1]
  elif work_type == 'children':
    df['work_type_Never_worked'] = [0]
    df['work_type_Private'] = [0]
    df['work_type_children'] = [1]
    df['work_type_Self-employed'] = [0]
  else:
    df['work_type_Never_worked'] = [0]
    df['work_type_Private'] = [0]
    df['work_type_children'] = [0]
    df['work_type_Self-employed'] = [0]


def residence_type_transformer(df, Residence_type):
  if Residence_type == 'Urban':
    df['Residence_type_Urban'] = [1]
  else:
    df['Residence_type_Urban'] = [0]

def smoking_transformer(df, smoking_status):
  if smoking_status == 'formerly smoked':
    df['smoking_status_formerly smoked'] = [1]
    df['smoking_status_never smoked'] = [0]
    df['smoking_status_smokes'] = [0]
  elif smoking_status == 'never smoked':
    df['smoking_status_formerly smoked'] = [0]
    df['smoking_status_never smoked'] = [1]
    df['smoking_status_smokes'] = [0]
  elif smoking_status == 'Smokes':
    df['smoking_status_formerly smoked'] = [0]
    df['smoking_status_never smoked'] = [0]
    df['smoking_status_smokes'] = [1]
  else:
    df['smoking_status_formerly smoked'] = [0]
    df['smoking_status_never smoked'] = [0]
    df['smoking_status_smokes'] = [0]

def prediction(age, hypertension, heart_disease, avg_glucose_level, bmi, gender, married, Residence_type, work_type, smoking_status):
  cols = ['age', 'hypertension', 'heart_disease',  'avg_glucose_level', 'bmi', 'gender_Male',
       'gender_Other', 'ever_married_Yes','work_type_Never_worked', 'work_type_Private',
       'work_type_Self-employed', 'work_type_children', 'Residence_type_Urban',
       'smoking_status_formerly smoked', 'smoking_status_never smoked',
       'smoking_status_smokes']
  df = pd.DataFrame(columns= cols)
  gender_transformer(df, gender)
  hypertension_transformer(df, hypertension)
  heart_disease_transformer(df, heart_disease)
  marriage_transformer(df, married)
  worktype_transformer(df, work_type)
  residence_type_transformer(df, Residence_type)
  smoking_transformer(df, smoking_status)
  columns = ['age', 'avg_glucose_level','bmi']
  df[columns] = [age, avg_glucose_level, bmi]
  df[columns]=df[columns].astype(int)
  df[columns] = scaler.transform(df[columns])
  


 
    # Making predictions 
  prediction = model.predict(df)
     
  if prediction == 0:
    pred = "No Heart attack"
  else:
    pred = "Heart Attack"
  return pred
         
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:grey;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Heart Stroke Prediction ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    #age = st.number_input("Age")
    age = st.slider('Age', 0, 100, 30)
    hypertension = st.selectbox('Hypertension',("...", "Yes","No"))
    heart_disease = st.selectbox('Heart Disease',("...","Yes","No"))
    #avg_glucose_level = st.number_input("Average Glucose Level")
    avg_glucose_level = st.slider('Average Glucose Level',0, 300, 100 )
    #bmi = st.number_input("BMI")
    bmi = st.slider('BMI', 0, 100, 30)
    gender = st.selectbox('Gender',("...","Male","Female","Other"))
    married = st.selectbox('Ever Married?',("...","Yes","No"))
    Residence_type = st.selectbox('Residence Type',("...","Urban","Rural")) 
    work_type = st.selectbox('Work Type',("...","Never_worked","Private","Self-employed", "children", 'Goverment'))
    smoking_status = st.selectbox('Smoking Status',("...","Smokes","never smoked", "formerly smoked"))

    result =""

    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(age, hypertension, heart_disease, avg_glucose_level, bmi, gender, married, Residence_type, work_type, smoking_status)
        st.success('You will have {}'.format(result))
        #print(LoanAmount)
     
if __name__=='__main__': 
    main()
