import numpy as np
import pickle
import streamlit as st
import pandas as pd

loaded_model = pickle.load(open('model_cleaned.sav','rb'))

def main():
    X = pd.read_csv("cleaned_data.csv")
    st.title('Credit Risk Scoring')

    CODE_GENDER = st.selectbox(
        'Client\'s gender',('Female','Male'))
    'You selected:', CODE_GENDER

    if CODE_GENDER=="Female":
        CODE_GENDER=0
    else:
        CODE_GENDER=1

    NAME_EDUCATION_TYPE = st.selectbox(
        'Client\'s last education', ('Lower Secondary','Secondary','Incomplete Higher','Higher Education','Academic Degree'))
    'You selected:', NAME_EDUCATION_TYPE

    if NAME_EDUCATION_TYPE=="Academic Degree":
        NAME_EDUCATION_TYPE=0
    elif NAME_EDUCATION_TYPE=="Higher Education":
        NAME_EDUCATION_TYPE=1
    elif NAME_EDUCATION_TYPE=="Incomplete Higher":
        NAME_EDUCATION_TYPE=2
    elif NAME_EDUCATION_TYPE=="Lower Secondary":
        NAME_EDUCATION_TYPE=3
    else:
        NAME_EDUCATION_TYPE=4

    FLAG_OWN_CAR = st.selectbox(
        'Client\'s car ownership',
        ('Owns a Car','Does not own a car'))
    'You selected:', FLAG_OWN_CAR

    if FLAG_OWN_CAR=="Does not own a car":
        FLAG_OWN_CAR=0
    else:
        FLAG_OWN_CAR=1

    REG_CITY_NOT_LIVE_CITY = st.selectbox(
        'Client\'s Permanent Address (City Level)',
        ('Different with Work Address','Same with Work Address'))
    'You selected:', REG_CITY_NOT_LIVE_CITY
    if REG_CITY_NOT_LIVE_CITY=="Same with Work Address":
        REG_CITY_NOT_LIVE_CITY=0
    else:
        REG_CITY_NOT_LIVE_CITY=1

    NAME_INCOME_TYPE = st.selectbox(
        'Client\'s source of income',
        ('Working','Commercial Associate','Pensioner','Stat Servant','Unemployed','Maternity Leave','Student','Businessman'))
    'You selected:', NAME_INCOME_TYPE

    if NAME_INCOME_TYPE=="Businessman":
        NAME_INCOME_TYPE=0
    elif NAME_INCOME_TYPE=="Commercial Associate":
        NAME_INCOME_TYPE=1
    elif NAME_INCOME_TYPE=="Maternity Leave":
        NAME_INCOME_TYPE=2
    elif NAME_INCOME_TYPE=="Pensioner":
        NAME_INCOME_TYPE=3
    elif NAME_INCOME_TYPE=="Stat Servant":
        NAME_INCOME_TYPE=4
    elif NAME_INCOME_TYPE=="Student":
        NAME_INCOME_TYPE=5
    elif NAME_INCOME_TYPE=="Unemployed":
        NAME_INCOME_TYPE=6
    else:
        NAME_INCOME_TYPE=7

    NAME_CONTRACT_TYPE = st.selectbox(
        'Clien\’s type of loans',
        ('Cash Loans','Revolving Loans'))
    'You selected:', NAME_CONTRACT_TYPE
    if NAME_CONTRACT_TYPE=="Cash Loans":
        NAME_CONTRACT_TYPE=0
    else:
        NAME_CONTRACT_TYPE=1


    REGION_RATING_CLIENT_W_CITY = st.selectbox(
        'Client\'s region rating',
        (1,2,3))
    'You selected:',REGION_RATING_CLIENT_W_CITY

    DAYS_BIRTH = st.text_input('Client\'s age this year')

    AMT_ANNUITY = st.text_input('Clien\’s loan annuity')

    AMT_GOODS_PRICE = st.text_input('Amount of client want to loan')

    diagnose = ''

    if st.button('Credit Risk Result'):
        diagnose = loan_predict([CODE_GENDER,NAME_EDUCATION_TYPE,FLAG_OWN_CAR,AMT_GOODS_PRICE,REG_CITY_NOT_LIVE_CITY,NAME_INCOME_TYPE, NAME_CONTRACT_TYPE, REGION_RATING_CLIENT_W_CITY, DAYS_BIRTH, AMT_ANNUITY])
    
    st.success(diagnose)

def loan_predict(input_data):
    input_array = np.asarray(input_data)
    re=input_array.reshape(1,-1)
    prediction=loaded_model.predict(re)
    print(prediction)
    
    if prediction[0] == 1:
        return 'This client can\'t pay the loan.'
    else:
        return 'This client can pay the loan.'

if __name__ == '__main__':
    main()
