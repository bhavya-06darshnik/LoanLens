
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from loan_classification_variables import LoanPred, ThresholdMetrics
import shap

import h2o


import pandas as pd
import joblib
import os
import re

import preprocess

LoanPredApp = FastAPI()


current_dir = os.getcwd()
model_filename = r'D:\LoanLens\app\model\dl_grid_model_66'
knn_initial_filename = r'D:\LoanLens\app\model\knn_imputer_model.pkl'
knn_cur_filename = r'D:\LoanLens\app\model\knn_imputer_model_no_multicol.pkl'
gbm_cur_filename=r'D:\LoanLens\app\model\gbm_grid1_model_155'
scaler_filename = r'D:\LoanLens\app\model\scaler_no_multicol.pkl'
purpose_filename = r'D:\LoanLens\app\model\purpose_mapping.pkl'

model_path = os.path.join(current_dir, model_filename)
knn_initial_path = os.path.join(current_dir, knn_initial_filename)
knn_cur_path = os.path.join(current_dir, knn_cur_filename)
scaler_path = os.path.join(current_dir, scaler_filename)
purpose_path = os.path.join(current_dir, purpose_filename)


h2o.init()

model = h2o.load_model(model_path)
gbm_model = h2o.load_model(gbm_cur_filename)
knn_initial_model = joblib.load(knn_initial_path)
knn_cur_model = joblib.load(knn_cur_path)
scaler = joblib.load(scaler_path)
purpose_mapping = joblib.load(purpose_path)

@LoanPredApp.get('/')
def index():

    return{'message': 'use /predict to make loan prediction'}

@LoanPredApp.post('/predict')
def predict_price(data: LoanPred, threshold: ThresholdMetrics):

    data = data.dict()


    df = preprocess.create_dataframe(data, knn_initial_model, purpose_mapping, knn_cur_model, scaler)


    hf = h2o.H2OFrame(df)


    threshold_metrics = threshold.threshold_metrics
    prediction = preprocess.predict_data(model, hf, threshold_metrics=threshold_metrics)

    return JSONResponse(content=prediction)

@LoanPredApp.post('/predict_explaination')
def predict_price(data: LoanPred, threshold: ThresholdMetrics):

    data = data.dict()

    df = preprocess.create_dataframe(
        data, knn_initial_model, purpose_mapping, knn_cur_model, scaler
    )

    hf = h2o.H2OFrame(df)

    threshold_metrics = threshold.threshold_metrics
    prediction = preprocess.predict_data(model, hf, threshold_metrics=threshold_metrics)


    feature_influence = preprocess.explain_prediction(gbm_model, hf)

    return JSONResponse(content={
        "prediction": prediction,
        "feature_influence": feature_influence
    })

if __name__ == '__main__':
    uvicorn.run("main:LoanPredApp",host='0.0.0.0', port=5000)
