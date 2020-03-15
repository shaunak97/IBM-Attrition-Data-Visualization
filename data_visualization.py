#1. Importing Libraries
from sklearn import model_selection
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import xgboost as xgb
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
from scipy import stats
from scipy.stats import norm, skew 
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) 

#2. Reading Data and Feature Engineering
data=pd.read_csv("IBM.csv")
data.drop("Over18", axis = 1,inplace=True)
data.drop("StandardHours", axis = 1,inplace=True)
data.drop("EmployeeNumber", axis = 1,inplace=True)
features=data.drop("Attrition", axis = 1)
target=data['Attrition']
target.replace('Yes',1,inplace=True) 
target.replace('No',0,inplace=True)
data["TotalSatisfaction"] = 'default value'
for index,data in data.iterrows():
    data["TotalSatisfaction"] = (data["EnvironmentSatisfaction"]+data["JobInvolvement"]+data["JobSatisfaction"]+data["WorkLifeBalance"]+data["RelationshipSatisfaction"])/5.0
=
data.drop("EnvironmentSatisfaction",inplace=True)
data.drop("JobSatisfaction",inplace=True)
data.drop("JobInvolvement",inplace=True)
data.drop("RelationshipSatisfaction",inplace=True)
data.drop("WorkLifeBalance",inplace=True)

#3. Creating Test Model
features1 = pd.get_dummies(features)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#numerical = ['Age','DailyRate','DistanceFromHome','Education','EnvironmentSatisfaction','HourlyRate','JobInvolvement','JobLevel','JobRole','JobSatisfaction','MonthlyIncome','MonthlyRate','NumCompaniesWorked','OverTime','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']
features1 = scaler.fit_transform(features1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features1, target, test_size=0.1, random_state=42)
from sklearn import svm
svm1 = svm.SVC()
svm1.fit(X_train, y_train)  
predictionssvm = svm1.predict(X_test)
dtc=DecisionTreeClassifier()
modeldtc = dtc.fit(X_train, y_train)
predictionsdtc = dtc.predict(X_test)
adb=AdaBoostClassifier()
modeladb=adb.fit(X_train, y_train)
predictionsadb = adb.predict(X_test)
from sklearn.ensemble import GradientBoostingClassifier
xgb= GradientBoostingClassifier()
modelxbg=xgb.fit(X_train, y_train)
predictionsxgb = xgb.predict(X_test)
import operator
from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(solver='adam',activation='tanh',random_state=0)
modelmlp=mlp.fit(X_train,y_train)
predictionmlp=mlp.predict(X_test)

#4. Stacked Classifier
X=features1
y=target
clf1 = adb
clf2 = dtc
clf3 = svm1
meta = LogisticRegression()

sclf = StackingClassifier(classifiers=[meta, clf1, clf3], 
                          meta_classifier=clf2)

for clf, label in zip([clf1, clf2, clf3, sclf], ['clf1','clf2','clf3','SC']):
    scores = model_selection.cross_val_score(clf, X, y, cv=6, scoring='accuracy')

#5. Data Visualization 
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools
gs = gridspec.GridSpec(2, 2)
fig = plt.figure(figsize=(10,8))
for clf, lab, grd in zip([clf1, clf2, clf3, sclf], 
    ['AdaBoost', 'Decision Tree Classifier', 'Support Vector Machine', 'Stacking Classifier'],itertools.product([0, 1], repeat=2)):

    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X_test, y=y_test, clf=clf)
    plt.title(lab)

model = Sequential()
model.add(Dense(200, input_dim=220, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adadelta()) # Mean squared error
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=32, verbose=2)

y_pred = model.predict(X_test)
plt.style.use('ggplot')
plt.plot(y_pred, y_test, 'ro')
plt.xlabel('Predictions', fontsize = 15)
plt.ylabel('Reality', fontsize = 15)
plt.title('Predictions x Reality on dataset Test', fontsize = 15)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.show()

