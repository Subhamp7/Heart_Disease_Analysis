# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 18:56:32 2020
@author: subham
"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__,static_url_path='/static')
model = pickle.load(open('LR_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
   
    final_features = (np.array(int_features)).reshape(1,-1)
    prediction = model.predict(final_features)
    if(prediction==0):
        output='Hurray!!! You are Safe'
    else:
        output='OOPS!!! You need medical attention'
    return render_template('index_predict.html', prediction_text=output)


if __name__ == "__main__":
    app.run(debug=True)