import numpy as np
import pickle
import streamlit as st
import pandas as pd

loaded_model = pickle.load(open('model.sav','rb'))

def main():
    train = pd.read_csv("application_train.csv")
    st.title('Credit Risk Scoring')

    train.drop('COMMONAREA_MEDI', axis = 'columns', inplace = True)
    train.drop('COMMONAREA_AVG', axis = 'columns', inplace = True)
    train.drop('COMMONAREA_MODE', axis = 'columns', inplace = True)
    train.drop('NONLIVINGAPARTMENTS_MODE', axis = 'columns', inplace = True)
    train.drop('NONLIVINGAPARTMENTS_AVG', axis = 'columns', inplace = True)
    train.drop('NONLIVINGAPARTMENTS_MEDI', axis = 'columns', inplace = True)
    train.drop('FONDKAPREMONT_MODE', axis = 'columns', inplace = True)
    train.drop('LIVINGAPARTMENTS_MODE', axis = 'columns', inplace = True)
    train.drop('LIVINGAPARTMENTS_AVG', axis = 'columns', inplace = True)
    train.drop('LIVINGAPARTMENTS_MEDI', axis = 'columns', inplace = True)
    train.drop('FLOORSMIN_AVG', axis = 'columns', inplace = True)
    train.drop('FLOORSMIN_MODE', axis = 'columns', inplace = True)
    train.drop('FLOORSMIN_MEDI', axis = 'columns', inplace = True)
    train.drop('YEARS_BUILD_MEDI', axis = 'columns', inplace = True)
    train.drop('YEARS_BUILD_MODE', axis = 'columns', inplace = True)
    train.drop('YEARS_BUILD_AVG', axis = 'columns', inplace = True)
    train.drop('OWN_CAR_AGE', axis = 'columns', inplace = True)
    train.drop('LANDAREA_AVG', axis = 'columns', inplace = True)

    train.drop('LANDAREA_MEDI', axis = 'columns', inplace = True)
    train.drop('LANDAREA_MODE', axis = 'columns', inplace = True)
    train.drop('BASEMENTAREA_MEDI', axis = 'columns', inplace = True)
    train.drop('BASEMENTAREA_AVG', axis = 'columns', inplace = True)
    train.drop('BASEMENTAREA_MODE', axis = 'columns', inplace = True)
    train.drop('EXT_SOURCE_1', axis = 'columns', inplace = True)
    train.drop('NONLIVINGAREA_MODE', axis = 'columns', inplace = True)
    train.drop('NONLIVINGAREA_AVG', axis = 'columns', inplace = True)
    train.drop('NONLIVINGAREA_MEDI', axis = 'columns', inplace = True)
    train.drop('APARTMENTS_AVG', axis = 'columns', inplace = True)
    train.drop('APARTMENTS_MEDI', axis = 'columns', inplace = True)
    train.drop('APARTMENTS_MODE', axis = 'columns', inplace = True)
    train.drop('ENTRANCES_AVG', axis = 'columns', inplace = True)
    train.drop('ENTRANCES_MODE', axis = 'columns', inplace = True)
    train.drop('ENTRANCES_MEDI', axis = 'columns', inplace = True)

    train.drop('LIVINGAREA_MODE', axis = 'columns', inplace = True)
    train.drop('LIVINGAREA_MEDI', axis = 'columns', inplace = True)
    train.drop('LIVINGAREA_AVG', axis = 'columns', inplace = True)
    train.drop('FLOORSMAX_MEDI', axis = 'columns', inplace = True)
    train.drop('FLOORSMAX_MODE', axis = 'columns', inplace = True)
    train.drop('FLOORSMAX_AVG', axis = 'columns', inplace = True)
    train.drop('YEARS_BEGINEXPLUATATION_MEDI', axis = 'columns', inplace = True)
    train.drop('YEARS_BEGINEXPLUATATION_MODE', axis = 'columns', inplace = True)
    train.drop('YEARS_BEGINEXPLUATATION_AVG', axis = 'columns', inplace = True)
    train.drop('TOTALAREA_MODE', axis = 'columns', inplace = True)

    train.drop('ELEVATORS_AVG', axis = 'columns', inplace = True)
    train.drop('ELEVATORS_MODE', axis = 'columns', inplace = True)
    train.drop('ELEVATORS_MEDI', axis = 'columns', inplace = True)

    train["OBS_30_CNT_SOCIAL_CIRCLE"] = train['OBS_30_CNT_SOCIAL_CIRCLE'].fillna(train['OBS_30_CNT_SOCIAL_CIRCLE'].median()) 
    train["DEF_30_CNT_SOCIAL_CIRCLE"] = train['DEF_30_CNT_SOCIAL_CIRCLE'].fillna(train['DEF_30_CNT_SOCIAL_CIRCLE'].median()) 
    train["OBS_60_CNT_SOCIAL_CIRCLE"] = train['OBS_60_CNT_SOCIAL_CIRCLE'].fillna(train['OBS_60_CNT_SOCIAL_CIRCLE'].median()) 
    train["DEF_60_CNT_SOCIAL_CIRCLE"] = train['DEF_60_CNT_SOCIAL_CIRCLE'].fillna(train['DEF_60_CNT_SOCIAL_CIRCLE'].median()) 
    train["EXT_SOURCE_2"] = train['EXT_SOURCE_2'].fillna(train['EXT_SOURCE_2'].median()) 
    train["AMT_GOODS_PRICE"] = train['AMT_GOODS_PRICE'].fillna(train['AMT_GOODS_PRICE'].median()) 
    train["AMT_ANNUITY"] = train['AMT_ANNUITY'].fillna(train['AMT_ANNUITY'].median()) 
    train["CNT_FAM_MEMBERS"] = train['CNT_FAM_MEMBERS'].fillna(train['CNT_FAM_MEMBERS'].median()) 
    train["DAYS_LAST_PHONE_CHANGE"] = train['DAYS_LAST_PHONE_CHANGE'].fillna(train['DAYS_LAST_PHONE_CHANGE'].median()) 

    train["NAME_TYPE_SUITE"] = train["NAME_TYPE_SUITE"].fillna(train["NAME_TYPE_SUITE"].mode()[0])
    train["OCCUPATION_TYPE"] = train["OCCUPATION_TYPE"].fillna(train["OCCUPATION_TYPE"].mode()[0])
    train["HOUSETYPE_MODE"] = train["HOUSETYPE_MODE"].fillna(train["HOUSETYPE_MODE"].mode()[0])
    train["ORGANIZATION_TYPE"] = train["ORGANIZATION_TYPE"].fillna(train["ORGANIZATION_TYPE"].mode()[0])
    train["WALLSMATERIAL_MODE"] = train["WALLSMATERIAL_MODE"].fillna(train["WALLSMATERIAL_MODE"].mode()[0])
    train["EMERGENCYSTATE_MODE"] = train["EMERGENCYSTATE_MODE"].fillna(train["EMERGENCYSTATE_MODE"].mode()[0])

    import scipy.stats as stats
    train_clean15 = train[(np.abs(stats.zscore(train["CNT_CHILDREN"])) < 3)] 
    train_clean14 = train_clean15[(np.abs(stats.zscore(train_clean15["AMT_INCOME_TOTAL"])) < 3)]
    train_clean13 = train_clean14[(np.abs(stats.zscore(train_clean14["AMT_CREDIT"])) < 3)]
    train_clean12 = train_clean13[(np.abs(stats.zscore(train_clean13["AMT_ANNUITY"])) < 3)]
    train_clean11 = train_clean12[(np.abs(stats.zscore(train_clean12["AMT_GOODS_PRICE"])) < 3)]
    train_clean10 = train_clean11[(np.abs(stats.zscore(train_clean11["REGION_POPULATION_RELATIVE"])) < 3)]
    train_clean9 = train_clean10[(np.abs(stats.zscore(train_clean10["DAYS_REGISTRATION"])) < 3)]
    train_clean8 = train_clean9[(np.abs(stats.zscore(train_clean9["CNT_FAM_MEMBERS"])) < 3)] 
    train_clean7 = train_clean8[(np.abs(stats.zscore(train_clean8["HOUR_APPR_PROCESS_START"])) < 3)] 
    train_clean6 = train_clean7[(np.abs(stats.zscore(train_clean7["EXT_SOURCE_2"])) < 3)] 
    train_clean5 = train_clean6[(np.abs(stats.zscore(train_clean6["DAYS_LAST_PHONE_CHANGE"])) < 3)] 
    train_clean4 = train_clean5[(np.abs(stats.zscore(train_clean5["OBS_30_CNT_SOCIAL_CIRCLE"])) < 3)] 
    train_clean3 = train_clean4[(np.abs(stats.zscore(train_clean4["OBS_60_CNT_SOCIAL_CIRCLE"])) < 3)] 
    train_clean2 = train_clean3[(np.abs(stats.zscore(train_clean3["DEF_60_CNT_SOCIAL_CIRCLE"])) < 3)] 
    train_clean1 = train_clean2[(np.abs(stats.zscore(train_clean2["DEF_30_CNT_SOCIAL_CIRCLE"])) < 3)] 

    from sklearn.utils import resample

    train_majority = train_clean1[(train_clean1['TARGET']== 0 )] #Majority
    train_minority = train_clean1[(train_clean1['TARGET']== 1 )] #Minority

    train_minority_upsampled = resample(train_minority, 
                                    replace=True,    # sample with replacement
                                    n_samples= 245119, # to match majority class
                                    random_state=42)  # reproducible results

    train_upsampled = pd.concat([train_minority_upsampled, train_majority])

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    train_upsampled['NAME_CONTRACT_TYPE']=le.fit_transform(train_upsampled['NAME_CONTRACT_TYPE'])
    train_upsampled['CODE_GENDER']=le.fit_transform(train_upsampled['CODE_GENDER'])
    train_upsampled['FLAG_OWN_CAR']=le.fit_transform(train_upsampled['FLAG_OWN_CAR'])
    train_upsampled['FLAG_OWN_REALTY']=le.fit_transform(train_upsampled['FLAG_OWN_REALTY'])
    train_upsampled['NAME_TYPE_SUITE']=le.fit_transform(train_upsampled['NAME_TYPE_SUITE'])
    train_upsampled['NAME_INCOME_TYPE']=le.fit_transform(train_upsampled['NAME_INCOME_TYPE'])
    train_upsampled['NAME_EDUCATION_TYPE']=le.fit_transform(train_upsampled['NAME_EDUCATION_TYPE'])
    train_upsampled['NAME_FAMILY_STATUS']=le.fit_transform(train_upsampled['NAME_FAMILY_STATUS'])
    train_upsampled['NAME_HOUSING_TYPE']=le.fit_transform(train_upsampled['NAME_HOUSING_TYPE'])
    train_upsampled['OCCUPATION_TYPE']=le.fit_transform(train_upsampled['OCCUPATION_TYPE'])
    train_upsampled['WEEKDAY_APPR_PROCESS_START']=le.fit_transform(train_upsampled['WEEKDAY_APPR_PROCESS_START'])
    train_upsampled['ORGANIZATION_TYPE']=le.fit_transform(train_upsampled['ORGANIZATION_TYPE'])
    train_upsampled['WALLSMATERIAL_MODE']=le.fit_transform(train_upsampled['WALLSMATERIAL_MODE'])
    train_upsampled['HOUSETYPE_MODE']=le.fit_transform(train_upsampled['HOUSETYPE_MODE'])
    train_upsampled['EMERGENCYSTATE_MODE']=le.fit_transform(train_upsampled['EMERGENCYSTATE_MODE'])

    X = train_upsampled[['CODE_GENDER','NAME_EDUCATION_TYPE','FLAG_OWN_CAR','AMT_GOODS_PRICE','REG_CITY_NOT_LIVE_CITY','NAME_INCOME_TYPE', 'NAME_CONTRACT_TYPE', 'REGION_RATING_CLIENT_W_CITY', 'DAYS_BIRTH', 'AMT_ANNUITY']]
    y = train_upsampled["TARGET"]

    CODE_GENDER = st.selectbox(
        'Client\'s Gender',('Female','Male'))
    'You selected:', CODE_GENDER

    if CODE_GENDER=="Female":
        CODE_GENDER=0
    else:
        CODE_GENDER=1

    NAME_EDUCATION_TYPE = st.selectbox(
        'Client\s Last Education ?', ('Lower Secondary','Secondary','Incomplete Higher','Higher Education','Academic Degree'))
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
        'Client Has A Car',
        ('Yes','No'))
    'You selected:', FLAG_OWN_CAR

    if FLAG_OWN_CAR=="No":
        FLAG_OWN_CAR=0
    else:
        FLAG_OWN_CAR=1

    REG_CITY_NOT_LIVE_CITY = st.selectbox(
        'Client Permanent Address ( City Level )',
        ('Different with Work Address','Same with Work Address'))
    'You selected:', REG_CITY_NOT_LIVE_CITY
    if REG_CITY_NOT_LIVE_CITY=="Same with Work Address":
        REG_CITY_NOT_LIVE_CITY=0
    else:
        REG_CITY_NOT_LIVE_CITY=1

    NAME_INCOME_TYPE = st.selectbox(
        'Client\'s Income From ?',
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
        'Client\s Type Loan?',
        ('Cash Loans','Revolving Loans'))
    'You selected:', NAME_CONTRACT_TYPE
    if NAME_CONTRACT_TYPE=="Cash Loans":
        NAME_CONTRACT_TYPE=0
    else:
        NAME_CONTRACT_TYPE=1


    REGION_RATING_CLIENT_W_CITY = st.selectbox(
        'Client Region Live?',
        (1,2,3))
    'You selected:',REGION_RATING_CLIENT_W_CITY

    DAYS_BIRTH = st.text_input('Client\s Age This Year')

    AMT_ANNUITY = st.slider('Amount of Client\'s Income In A Year',int(X.AMT_ANNUITY.min()),int(X.AMT_ANNUITY.max()),int(X.AMT_ANNUITY.mean()))

    AMT_GOODS_PRICE = st.slider('Amount of Client Want To Loan',int(X.AMT_GOODS_PRICE.min()),int(X.AMT_GOODS_PRICE.max()),int(X.AMT_GOODS_PRICE.mean()))

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
        return 'This Client Can\'t Pay the Loan'
    else:
        return 'This Client Can Pay the Loan'

if __name__ == '__main__':
    main()
