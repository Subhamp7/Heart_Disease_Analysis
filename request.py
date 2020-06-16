import requests

url = "http://localhost:5000/predict"
r = requests.post(url,json={"Age":45, 
                            "Gender":1,
                            "Chest pain type":3 ,
			                "Resting blood pressure":120, 
                            "Serum cholestoral in mg/dl":110, 
                            "Fasting blood sugar":1,
                			"Resting electrocardiographic results":1, 
                            "Maximum heart rate achieved":130, 
                            "Exercise induced angina":1 ,
                            "oldpeak = ST depression induced by exercise relative to rest":26,
                            "The slope of the peak exercise ST segment":1,
                            "Number of major vessels colored by flourosopy":3,
                            "Thal":1,
                            "Diagnosis of heart disease (angiographic disease status)":0})

print(r.json())