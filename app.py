from flask import Flask, request, render_template, redirect
import pickle
import pandas as pd
import numpy as np
import lightgbm
from shapash.explainer.smart_explainer import SmartExplainer

app = Flask(__name__)

df = pd.read_csv('clients.csv', index_col=0)
xpl = SmartExplainer()
xpl.load("xpl.pickle")
xpl.predict()
xpl.predict_proba()
dataCustomer = xpl.x_pred

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict")
def search():
    user_input = request.args.get("id", type=int)
    if user_input not in df['SK_ID_CURR'].unique():
        response = {"ERROR" : f"{user_input} is not in our list of clients'"}
        return response
    idx = dataCustomer[df['SK_ID_CURR'] == int(user_input)].index[0]
    predExact = xpl.y_pred[xpl.y_pred.index.isin([idx])]['pred'].values[0]
    predProba = xpl.proba_values[xpl.proba_values.index.isin([idx])]['class_0.0'].values[0]
    response = {'Decision': "Loan Granted!" if not predExact else "Loan Refused!",
                'Repayment Probability':f"{predProba*100:.1f}%"}
    
    return response


if __name__ == "__main__":
    app.run()