#%%
""" Libraries
================================= """

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix

#%%
""" Reading the data
================================= """

# Data paths
TRAIN_PATH = "dataset/Training.csv"
TEST_PATH = "./dataset/Testing.csv"

# Reading the train.csv by removing the last column since it's an empty column
data = pd.read_csv(TRAIN_PATH).dropna(axis = 1)
test_data = pd.read_csv(TEST_PATH).dropna(axis=1)

# Checking whether the dataset is balanced or not
disease_counts = data["prognosis"].value_counts()
temp_df = pd.DataFrame({
	"Disease": disease_counts.index,
	"Counts": disease_counts.values
})

plt.figure(figsize = (8,18))
sns.barplot(x = "Counts", y = "Disease", data = temp_df)
plt.show()

countsn = len(np.unique(temp_df['Counts']))
print(f"The number of Diseases data are{' ' if countsn==1 else ' NOT '}balanced")

# Encoding the target value into numerical value
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

#Splitting the data for training and testing the model
X = data.iloc[:,:-1]
y = data.iloc[:, -1]
X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size = 0.2, random_state = 42)

#%% K-Fold Cross-Validation

def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))

# Initializing Models
models = {
	"SV": SVC(),
	"NB": GaussianNB(),
	"RF": RandomForestClassifier(random_state=18)
}

# Producing cross validation score for the models
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
print("=="*30)
for i,model_name in enumerate(models):
	model = models[model_name]
	scores = cross_val_score(model, X, y, cv = cv, 
							n_jobs = -1, 
                            scoring = cv_scoring)
	if i>0:
		print("--"*30)
	print(model_name)
	#print(f"Scores: {scores}")
	print(f"Mean Score: {np.mean(scores):0.3f} +/- {np.std(scores):0.3f}")
print("=="*30)

#%% Training and testing

def printresults(model,X_train,y_train,y_cv,preds,model_name):
	
	print(f"Accuracy on train data by {model_name}\
	: {accuracy_score(y_train, model.predict(X_train))*100}")

	print(f"Accuracy on test data by {model_name}\
	: {accuracy_score(y_cv, preds)*100}")

	cf_matrix = confusion_matrix(y_cv, preds)

	plt.figure(figsize=(12,8))
	sns.heatmap(cf_matrix, annot=True)
	plt.title(f"Confusion Matrix for {model_name} Classifier on Cross Validation Data")
	plt.xlabel("Predicted")
	plt.ylabel("True")
	plt.show()

def runmodel(model,X_train,X_cv,y_train,y_cv,model_name):
	model.fit(X_train, y_train)
	preds = model.predict(X_cv)
	printresults(model,X_train,y_train,y_cv,preds,model_name)

for model_name in models:
	print('=='*30)
	runmodel(models[model_name],X_train,X_cv,y_train,y_cv,model_name)

#%% Reading the test data

X_test = test_data.iloc[:, :-1]
Y_test = encoder.transform(test_data.iloc[:, -1])

final_models = {
	"svm": SVC(),
	"nb" : GaussianNB(),
	"rf" : RandomForestClassifier(random_state=18)
	}

preds = {}
for m in final_models.keys():
	final_models[m].fit(X_train,y_train)
	preds[m] = final_models[m].predict(X_test)

final_preds = [mode([i,j,k]).mode for i,j,k in zip(preds['svm'], preds['nb'], preds['rf'])]

cf_matrix = confusion_matrix(Y_test, final_preds)
plt.figure(figsize=(12,8))
 
sns.heatmap(cf_matrix, annot = True)
plt.title("Confusion Matrix for Combined Model on Test Dataset")
plt.show()
#%%