#import all necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

from google.colab import files

files1 = files.upload()

for filename, content in files1.items():
  with open(filename, 'wb') as f:
    f.write(content)

#create variable to house data
data = pd.read_csv("diabetes.csv")
data.head()

print(len(data))

#check for data is null
data['BMI'].replace(0,data['BMI'].mean())
data['BloodPressure'] = data['BloodPressure'].replace(0,data['BloodPressure'].mean())
data['Glucose'] = data['Glucose'].replace(0,data['Glucose'].mean())
data['Insulin'] = data['Insulin'].replace(0,data['Insulin'].mean())
data['SkinThickness'] = data['SkinThickness'].replace(0,data['SkinThickness'].mean())
print(data.isna())
np.sum(data.isnull())
print(data.head())

data.info()

print(data.corr())
sns.heatmap(data.corr())

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principal_components = pca.fit_transform(data)

principal_df = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2'])

principal_df.tail()

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Diabetes Dataset",fontsize=20)
targets = [0, 1] # 0 means no diabetes, 1 means has diabetes
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = data['Outcome'] == target
    plt.scatter(principal_df.loc[indicesToKeep, 'PC1']
               , principal_df.loc[indicesToKeep, 'PC2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})


print(len(data))

X = data.drop(columns='Outcome')
Y = data['Outcome']

Xmatrix = X.to_numpy()
Ymatrix = Y.to_numpy()

print(Xmatrix)
print(Ymatrix)

x_train, x_test, y_train, y_test = train_test_split(Xmatrix, Ymatrix, test_size = 0.2, random_state = 2)


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#data visualization
print(data.describe())

plt.figure(figsize=(10,6))
plt.hist(data['BMI'], color='orange', edgecolor = 'black')
plt.xlabel('BMI')
plt.show()

plt.figure(figsize=(10,6))
plt.hist(data['Age'], color='green', edgecolor = 'black')
plt.xlabel('Age')
plt.show()

plt.figure(figsize=(10,6))
plt.hist(data['Insulin'], color='purple', edgecolor = 'black')
plt.xlabel('Insulin')
plt.show()

plt.figure(figsize=(10,6))
plt.hist(data['Glucose'], color='red', edgecolor = 'black')
plt.xlabel('Glucose')
plt.show()

plt.figure(figsize=(10,6))
plt.hist(data['Pregnancies'], color='blue', edgecolor = 'black')
plt.xlabel('Pregnancies')
plt.show()

plt.figure(figsize=(10,6))
plt.hist(data['BloodPressure'], color='yellow', edgecolor = 'black')
plt.xlabel('BloodPressure')
plt.show()

plt.figure(figsize=(10,6))
plt.hist(data['DiabetesPedigreeFunction'], color='black', edgecolor = 'white')
plt.xlabel('DiabetesPedigreeFunction')
plt.show()

plt.figure(figsize=(10,6))
plt.hist(data['SkinThickness'], color='pink', edgecolor = 'black')
plt.xlabel('SkinThickness')
plt.show()

#Boxplots

plt.figure(figsize=(10,6))
sns.boxplot(x=data['BMI'])

plt.figure(figsize=(10,6))
sns.boxplot(x=data['Age'])

plt.figure(figsize=(10,6))
sns.boxplot(x=data['Insulin'])

plt.figure(figsize=(10,6))
sns.boxplot(x=data['Glucose'])

plt.figure(figsize=(10,6))
sns.boxplot(x=data['SkinThickness'])

plt.figure(figsize=(10,6))
sns.boxplot(x=data['Pregnancies'])

plt.figure(figsize=(10,6))
sns.boxplot(x=data['DiabetesPedigreeFunction'])

plt.figure(figsize=(10,6))
sns.boxplot(x=data['BloodPressure'])

#Baseline model development

#logisticRegression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report


instance = LogisticRegression()
instance.fit(x_train, y_train)

pred = instance.predict(x_test)
print("Model metrics of logistic regression")
print("Accuracy score:", accuracy_score(y_test, pred))
print("f1_score:", f1_score(y_test, pred))
print("recall_score", recall_score(y_test, pred))
print("precision_score", precision_score(y_test, pred))

print(classification_report(y_test, pred))

#RidgeClassifier

from sklearn.linear_model import RidgeClassifier

instance2 = RidgeClassifier()
instance2.fit(x_train, y_train)

pred2 = instance2.predict(x_test)
print("Model metrics of ridgeclassifier")
print("Accuracy score:", accuracy_score(y_test, pred2))
print("f1_score:", f1_score(y_test, pred2))
print("recall_score", recall_score(y_test, pred2))
print("precision_score", precision_score(y_test, pred2))
print(classification_report(y_test, pred2))

#RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

instance3 = RandomForestClassifier()
instance3.fit(x_train, y_train)

pred3 = instance3.predict(x_test)
print("Model metrics of randomforestclassifier")
print("Accuracy score:", accuracy_score(y_test, pred3))
print("f1_score:", f1_score(y_test, pred3))
print("recall_score", recall_score(y_test, pred3))
print("precision_score", precision_score(y_test, pred3))
print(classification_report(y_test, pred3))
#DecisionTrees

from sklearn.tree import DecisionTreeClassifier

instance4 = DecisionTreeClassifier()
instance4.fit(x_train,y_train)

pred4 = instance4.predict(x_test)
print("Model metrics of decisiontreeclassifier")
print("Accuracy score:", accuracy_score(y_test, pred4))
print("f1_score:", f1_score(y_test, pred4))
print("recall_score", recall_score(y_test, pred4))
print("precision_score", precision_score(y_test, pred4))
print(classification_report(y_test, pred4))
#Support Vector Machines

from sklearn.svm import SVC

instance5 = SVC()
instance5.fit(x_train, y_train)

pred5 = instance5.predict(x_test)
print("Model metrics of SVC")
print("Accuracy score:", accuracy_score(y_test, pred5))
print("f1_score:", f1_score(y_test, pred5))
print("recall_score", recall_score(y_test, pred5))
print("precision_score", precision_score(y_test, pred5))
print(classification_report(y_test, pred5))

#Using SVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc

confmatrix = confusion_matrix(y_test, pred5)
disp = ConfusionMatrixDisplay(confmatrix)
disp.plot()

fpr, tpr, thresholds = roc_curve(y_test, pred5)
disp2 = RocCurveDisplay.from_predictions(y_test, pred5)


plt.show()

print(auc(fpr, tpr))

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

p = []

for i in range(1,1000):
  instance5 = SVC(C = i, random_state = 1)
  instance5.fit(x_train, y_train)

  pred5 = instance5.predict(x_test)
  p.append(accuracy_score(y_test, pred5))
  print(f"Accuracy score: {accuracy_score(y_test, pred5)}")

plt.figure(figsize=(10,6))
plt.plot(range(1, 1000), p)

print()

g = [0.0001, 0.001, 0.01, 0.1]
a = []
for j in range(len(g)):
  instance5 = SVC(gamma = g[j])
  instance5.fit(x_train, y_train)

  pred5 = instance5.predict(x_test)
  print(f"Accuracy score: {accuracy_score(y_test, pred5)}")
  a.append(accuracy_score(y_test, pred5))


plt.figure(figsize=(10,6))
plt.plot(range(len(g)), a)

from sklearn.model_selection import GridSearchCV

# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf'],
              'cache_size': [1, 100, 200, 300, 400]}

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)


# fitting the model for grid search
grid.fit(x_train, y_train)


# print best parameter after tuning
print(grid.best_params_)

# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)


grid_predictions = grid.predict(x_test)

svctuned = SVC(C = 100, gamma = 0.0000001, kernel = 'rbf')
svctuned.fit(x_train, y_train)

svctunedPred = svctuned.predict(x_test)

print(f"Accuracy score: {accuracy_score(y_test, svctunedPred)}")
print(f"f1_score score: {f1_score(y_test, svctunedPred)}")
print(f"Recall score: {recall_score(y_test, svctunedPred)}")
print(f"Precision score: {precision_score(y_test, svctunedPred)}")

print(classification_report(y_test, svctunedPred))
