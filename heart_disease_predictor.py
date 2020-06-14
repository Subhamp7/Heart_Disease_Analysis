# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 19:31:10 2020
@author: subham
"""
#importing the libraries
import pandas as pd
import numpy  as np
from sklearn.preprocessing    import OneHotEncoder
from sklearn.compose          import ColumnTransformer
from sklearn.preprocessing    import StandardScaler
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.linear_model     import LogisticRegression
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.svm              import SVC
import warnings
import pickle

warnings.filterwarnings('ignore')

#loading the dataset
dataset=pd.read_csv('heart_data.csv', sep='\t' )

#to check the diferrent unique values in the dataset
for index in dataset.columns:
    print(index,dataset[index].unique())
print(dataset.dtypes)
print('The number of missing dataset:', dataset.isnull().sum())
    
#splitting the dataset to independent and dependent sets
dataset_X=dataset.iloc[:,  0:13].values
dataset_Y=dataset.iloc[:, 13:14].values

#columns to be encoded: cp(2), restecg(6), slope(10), ca(11), thal(12)
#ct=ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [2,6,10,11,12])], remainder='passthrough')
#dataset_X = np.array(ct.fit_transform(dataset_X), dtype = np.str)

#scaling the dataset
sc=StandardScaler()
dataset_X=sc.fit_transform(dataset_X)

#splitting data to training set and test set
X_train, X_test, Y_train, Y_test =train_test_split(dataset_X, dataset_Y, test_size=0.25 , random_state=0)

#scores
def scores(pred,test,model):
    print(('\n==========Scores for {} ==========\n').format(model))
    print(f"Accuracy Score   : {accuracy_score(pred,test) * 100:.2f}% " )
    print(f"Precision Score  : {precision_score(pred,test) * 100:.2f}% ")
    print(f"Recall Score     : {recall_score(pred,test) * 100:.2f}% " )
    print("Confusion Matrix :\n" ,confusion_matrix(pred,test))
    
    
#logistic regression
lr=LogisticRegression(solver='liblinear')
lr.fit(X_train, Y_train)
Y_pred_lr=lr.predict(X_test)
scores(Y_pred_lr,Y_test,'Logistic_Regression')

#KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred_knn=knn.predict(X_test)
scores(Y_pred_knn,Y_test,'KNeighbors_Classifier')

#SVC
svm = SVC(kernel='rbf', gamma=0.1, C=1.0)
svm.fit(X_train, Y_train)
Y_pred_svm=svm.predict(X_test)
scores(Y_pred_svm,Y_test,'Support_Vector_machine')

#so the best is logistic regression
#saving model to disk
pickle.dump(lr, open('LR_model.pkl', 'wb'))


#test the pickle file
model=pickle.load(open('LR_model.pkl', 'rb'))
value=dataset_X[20,:].reshape(1,-1)
real=dataset_Y[20,:]
print(("\n The value predicted is : {} and the real value is : {} ").format(model.predict(value), real))

print('\nCompleted')