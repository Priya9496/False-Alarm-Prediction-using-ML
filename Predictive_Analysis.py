# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 23:11:43 2021

"""
##################  Logistic Regression  ######################

import pandas as pd
import numpy as np

default = pd.read_excel("E:/IACSD_DBDA_Online_Lec/project_code/False_Alarm/False_Alarm_Case.xlsx")
fac = default.drop('Case No.', axis =True)
dum_Default = pd.get_dummies(fac, drop_first=True)


X = dum_Default.iloc[:,0:6]
y = dum_Default.iloc[:,6:]

dum_Default['Spuriosity Index(0/1)'].value_counts() 

# Import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=2020)

# Create the classifier: logreg
logreg = LogisticRegression()
# Fit the classifier to the training data
logreg.fit(X_train,y_train)


# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)
# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred))

##################  Gaussian Naive Bayes  ######################

import pandas as pd
import numpy as np

default = pd.read_excel("E:/IACSD_DBDA_Online_Lec/project_code/False_Alarm/False_Alarm_Case.xlsx")
fac = default.drop('Case No.', axis =True)
dum_Default = pd.get_dummies(fac, drop_first=True)


X = dum_Default.iloc[:,0:6]
y = dum_Default.iloc[:,6:]

from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state=2020,
                                                    stratify=y)

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
y_pred = gaussian.predict(X_test)

# Model Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

##################  K-Nearest Neighbours  ######################

import pandas as pd
import numpy as np

default = pd.read_excel("E:/IACSD_DBDA_Online_Lec/project_code/False_Alarm/False_Alarm_Case.xlsx")
fac = default.drop('Case No.', axis =True)
dum_Default = pd.get_dummies(fac, drop_first=True)


X = dum_Default.iloc[:,0:6]
y = dum_Default.iloc[:,6:]


from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, 
                                                    random_state=2020,
                                                    stratify=y)
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit( X_train , y_train )
y_pred = knn.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

##################  Support Vector Machine  ######################

import pandas as pd
import numpy as np

default = pd.read_excel("E:/IACSD_DBDA_Online_Lec/project_code/False_Alarm/False_Alarm_Case.xlsx")
fac = default.drop('Case No.', axis =True)
dum_Default = pd.get_dummies(fac, drop_first=True)


X = dum_Default.iloc[:,0:6]
y = dum_Default.iloc[:,6:]

dum_Default['Spuriosity Index(0/1)'].value_counts() 

from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.svm import SVC



# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, 
                                                    random_state=2020,
                                                    stratify=y)

svc = SVC(probability = True,kernel='rbf')
fitSVC = svc.fit(X_train, y_train)
y_pred = fitSVC.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

##################  Decision Tree Classifier  ######################

import pandas as pd
import numpy as np

default = pd.read_excel("E:/IACSD_DBDA_Online_Lec/project_code/False_Alarm/False_Alarm_Case.xlsx")
fac = default.drop('Case No.', axis =True)
dum_Default = pd.get_dummies(fac, drop_first=True)


X = dum_Default.iloc[:,0:6]
y = dum_Default.iloc[:,6:]

dum_Default['Spuriosity Index(0/1)'].value_counts() 

# Import the necessary modules
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=2020,
                                                    stratify=y)

clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred))

##################  Random Forest Classifier  ######################

import pandas as pd
import numpy as np

default = pd.read_excel("E:/IACSD_DBDA_Online_Lec/project_code/False_Alarm/False_Alarm_Case.xlsx")
fac = default.drop('Case No.', axis =True)
dum_Default = pd.get_dummies(fac, drop_first=True)


X = dum_Default.iloc[:,0:6]
y = dum_Default.iloc[:,6:]

dum_Default['Spuriosity Index(0/1)'].value_counts() 

from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, 
                                                    random_state=2018,
                                                    stratify=y)

model_rf = RandomForestClassifier(random_state=2020,
                                  n_estimators=500,oob_score=True)
model_rf.fit( X_train , y_train )
y_pred = model_rf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

####################################################################