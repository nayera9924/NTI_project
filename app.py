import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Loan Default Prediction System")
st.write(
    "Enter the applicant's information below to predict the probability "
)

@st.cache_resource
def load_model():

    model = joblib.load("best_model.pkl")
    scaler = joblib.load("preprocessor.pkl")
    return model, scaler


try:
    model, scaler = load_model()

except Exception as e:
    st.error(
        "Could not load the model files. "
    )
    st.exception(e)
    st.stop()

MODEL_FEATURES = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio",

    "HasMortgage",
    "HasDependents",
    "HasCoSigner",

    "LoanToIncomeRatio",
    "EstMonthlyPayment",
    "EmploymentStability",
    "DisposableIncome",
    "RiskFlag",
    "PaymentToIncomeRatio",
    "TotalDebtBurden",
    "AgeAtPayoff",
    "IncomePerCreditLine",

    "HighEducation_FullTime",

    "Education_High School",
    "Education_Master's",
    "Education_PhD",

    "EmploymentType_Part-time",
    "EmploymentType_Self-employed",
    "EmploymentType_Unemployed",

    "MaritalStatus_Married",
    "MaritalStatus_Single",

    "LoanPurpose_Business",
    "LoanPurpose_Education",
    "LoanPurpose_Home",
    "LoanPurpose_Other",

    "CreditScoreBin_Fair",
    "CreditScoreBin_Good",
    "CreditScoreBin_Excellent"
]

def prepare_input(
    age,
    income,
    loan_amount,
    credit_score,
    months_employed,
    num_credit_lines,
    interest_rate,
    loan_term,
    dti_ratio,

    education,
    employment_type,
    marital_status,

    has_mortgage,
    has_dependents,
    loan_purpose,
    has_cosigner

):

    df = pd.DataFrame([{
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti_ratio,
        "Education": education,
        "EmploymentType": employment_type,
        "MaritalStatus": marital_status,
        "HasMortgage": has_mortgage,
        "HasDependents": has_dependents,
        "LoanPurpose": loan_purpose,
        "HasCoSigner": has_cosigner
    }])

    # FEATURE ENGINEERING
    df["LoanToIncomeRatio"] = (df["LoanAmount"] / df["Income"])
    df["EstMonthlyPayment"] = (df["LoanAmount"]* (df["InterestRate"] / 100)/ df["LoanTerm"])
    df["EmploymentStability"] = (df["MonthsEmployed"]/ (df["Age"] * 12))
    df["DisposableIncome"] = (df["Income"] * (1 - df["DTIRatio"]))

    bins = [-float("inf"),579,669,739,float("inf")]
    labels = ["Poor","Fair","Good","Excellent"]
    df["CreditScoreBin"] = pd.cut(df["CreditScore"],bins=bins,labels=labels)

    df["RiskFlag"] = ((df["HasCoSigner"] == "No")&(df["HasMortgage"] == "No")&(df["DTIRatio"] > 0.5)).astype(int)
    df["PaymentToIncomeRatio"] = (df["EstMonthlyPayment"]/(df["Income"] / 12))
    df["TotalDebtBurden"] = (df["DTIRatio"] + df["LoanToIncomeRatio"])
    df["AgeAtPayoff"] = (df["Age"]+(df["LoanTerm"] / 12))
    df["IncomePerCreditLine"] = (df["Income"] / (df["NumCreditLines"] + 1))
    df["HighEducation_FullTime"] = (df["Education"].isin(["Bachelor's","Master's","PhD"])&(df["EmploymentType"] == "Full-time")).astype(int)

    binary_columns = ["HasMortgage","HasDependents","HasCoSigner"]

    for column in binary_columns:
        df[column] = df[column].map({
            "Yes": 1,
            "No": 0
        })

    categorical_columns = ["Education","EmploymentType","MaritalStatus","LoanPurpose","CreditScoreBin"]

    df = pd.get_dummies( df,columns=categorical_columns,drop_first=True,dtype=int)

    df = df.reindex(columns=MODEL_FEATURES,fill_value=0)
    # Convert everything to float
    # df = df.astype(float)

    scaled_data = scaler.transform(df)
    return scaled_data

st.subheader("Applicant Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age",min_value=18,max_value=100,value=30)
    income = st.number_input("Annual Income ($)",min_value=1,value=50000)
    loan_amount = st.number_input("Loan Amount ($)",min_value=1000,value=10000)
    credit_score = st.slider("Credit Score",min_value=300,max_value=850,value=650)
    months_employed = st.number_input( "Months Employed", min_value=0, value=24)
    num_credit_lines = st.number_input("Number of Credit Lines", min_value=1, value=3)
    interest_rate = st.number_input("Interest Rate (%)",min_value=0.0, value=10.0)
    loan_term = st.number_input("Loan Term (Months)",min_value=12,max_value=360,value=36)

with col2:
    dti_ratio = st.slider("DTI Ratio",min_value=0.0, max_value=1.0, value=0.30)
    education = st.selectbox("Education Level",["High School","Bachelor's","Master's","PhD"])
    employment_type = st.selectbox( "Employment Type", ["Full-time","Part-time","Self-employed","Unemployed"])
    marital_status = st.selectbox("Marital Status", ["Single","Married","Divorced"])
    loan_purpose = st.selectbox("Loan Purpose",["Auto","Business","Education","Home","Other"])
    has_mortgage = st.radio("Has Mortgage?",["Yes","No"])
    has_dependents = st.radio("Has Dependents?",["Yes","No"])
    has_cosigner = st.radio("Has Co-Signer?",["Yes","No"])


st.markdown("---")
if st.button(" Predict Default Status",type="primary",use_container_width=True):
    try:
        processed_data = prepare_input(
            age=age,
            income=income,
            loan_amount=loan_amount,
            credit_score=credit_score,
            months_employed=months_employed,
            num_credit_lines=num_credit_lines,
            interest_rate=interest_rate,
            loan_term=loan_term,
            dti_ratio=dti_ratio,
            education=education,
            employment_type=employment_type,
            marital_status=marital_status,
            has_mortgage=has_mortgage,
            has_dependents=has_dependents,
            loan_purpose=loan_purpose,
            has_cosigner=has_cosigner

        )


        probability = model.predict_proba(processed_data)[0][1]
        threshold = 0.58
        prediction = int(probability >= threshold)

        st.markdown("---")
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(
                f"🚨 High Risk\n\n"
                f"Estimated probability of default: "
                f"**{probability:.1%}**"
            )

        else:
            st.success(
                f"✅ Low Risk\n\n"
                f"Estimated probability of default: "
                f"**{probability:.1%}**"
            )


        st.progress(
            float(probability)
        )


        st.write(
            f"Default probability: **{probability:.2%}**"
        )

        st.write(
            f"Decision threshold: **{threshold:.0%}**"
        )


    except Exception as e:
        st.error(
            "An error occurred while making the prediction."
        )
        st.exception(e)