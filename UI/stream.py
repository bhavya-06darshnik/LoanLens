import streamlit as st
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:8000/predict_explaination"

st.title("LoanLens – JSON Prediction Interface")

st.write("Paste the JSON request body below and run the model.")

default_json = """
{
  "data": {
    "current_loan_amount": 5167,
    "term": "Long Term",
    "credit_score": 350.0,
    "years_in_current_job": 10,
    "home_ownership": "Own Home",
    "annual_income": 127010.0,
    "purpose": "Debt Consolidation",
    "monthly_debt": 1061.51,
    "years_of_credit_history": 25.8,
    "months_since_last_delinquent": 5.5,
    "number_of_open_accounts": 7,
    "number_of_credit_problems": 0,
    "current_credit_balance": 112833,
    "maximum_open_credit": 16954.0,
    "bankruptcies": 0.0,
    "tax_liens": 0.0
  },
  "threshold": {
    "threshold_metrics": "precision"
  }
}
"""

json_input = st.text_area("Input JSON", default_json, height=350)

if st.button("Run Prediction"):

    try:
        payload = json.loads(json_input)

        response = requests.post(API_URL, json=payload)

        if response.status_code != 200:
            st.error(f"API error: {response.status_code}")
        else:
            result = response.json()

            prediction = result["prediction"]["prediction"][0]
            influence = result["feature_influence"]

            st.subheader("Prediction Result")
            st.success(prediction)

            df = pd.DataFrame(
                list(influence.items()),
                columns=["Feature", "Contribution"]
            )

            df = df.sort_values("Contribution", ascending=False)

            st.subheader("Feature Influence Table")
            st.dataframe(df)

            st.subheader("Feature Influence Graph")

            fig, ax = plt.subplots()

            colors = ["green" if x > 0 else "red" for x in df["Contribution"]]

            ax.barh(df["Feature"], df["Contribution"], color=colors)
            ax.set_xlabel("Contribution")
            ax.set_ylabel("Feature")
            ax.invert_yaxis()

            st.pyplot(fig)

    except Exception as e:
        st.error(str(e))