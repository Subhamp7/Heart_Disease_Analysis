# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 18:56:32 2020
@author: subham
"""
import pickle
import numpy 
from   flask import Flask, request, render_template


app   = Flask(__name__,static_url_path='/static')

#loading the models
model = pickle.load(open('ml_model.pkl', 'rb'))
enco  = pickle.load(open('encoder.pkl',  'rb'))
scal  = pickle.load(open('scaling.pkl',  'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    attibutes        = ["Age", "Gender","CP", "RBP", "SC", "FBS","RER", 
                        "MHR", "EIA", "ST", "SST", "Flourosopy", "Thal"]
    attibutes_val    = [float(request.form[items]) for items in attibutes]
    attibutes_array  = (numpy.array(attibutes_val)).reshape(1,-1)
    attibutes_encode = enco.transform(attibutes_array)
  

    prediction       = model.predict(attibutes_encode)
    
    if(prediction==1):
        output = 'OOPS!!! You need medical attention'
    else:
        output = 'Hurray!!! You are Safe'
    
        
    return render_template('index_predict.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)