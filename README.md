# Machine Learning - Disease Prediction

## Objective

This script aims to implement a robust machine-learning model using the [scikit-learn](https://scikit-learn.org/stable/) library that can efficiently predict the disease of a human, based on the symptoms that he/she possesses. 

## Introduction and Methods

### **The Data**

**Gathering:** The idea of this project came from a Kaggle problem (https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning). Therefore, I used the dataset from this problem. This dataset consists of two CSV files one for training and one for testing. There is a total of 133 columns in the dataset out of which 132 columns represent the symptoms and the last column is the prognosis.

**Cleaning:** After acquiring and observing the dataset, I noticed that all the columns' symptoms have binary information (the subject has or hasn’t the symptom), and the target column i.e. prognosis is a string type. To clean the data, I excluded any empty columns and encoded the target column to numerical form using a [label encoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html).

```python
data = pd.read_csv(TRAIN_PATH).dropna(axis = 1)
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])
```

### The Models

For this project, I used this cleaned data to train the **Support Vector Classifier (SV)**, **Naive Bayes Classifier (NB)**, and **Random Forest Classifier (RF)**. To implement the models I used the [sklearn](https://scikit-learn.org/stable/) library. Then, I used *confusion matrices* to determine the quality of the models. To improve the robustness and accuracy of the overall prediction, after training the three models, I predicted the disease for the input symptoms by combining the predictions of all three models. The combined model gives as a result the most frequent answer of the three models.

## Results

### **Reading the dataset**

Whenever solving a classification task it is necessary to check whether the target column is balanced or not. For that, I used a bar plot to check it. From Figure 1, we can observe that the dataset is a balanced dataset i.e. there are exactly 120 samples for each disease, and no further balancing is required.

![Figure 1. The number of entries for each disease (the target). ](https://raw.githubusercontent.com/LexC/ML_Disease_Prediction/main/outputs/databalance.png)

Figure 1. The number of entries for each disease (the target). 

### **K-Fold Cross-Validation**

After splitting the data training the data into 80:20 format, I used K-Fold cross-validation to evaluate the proposed machine-learning models (SV, NB, and RF). The implemented function for the K-Fold cross-validation was *[sklearn.model_selection.cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)*, as shown in the code bellow:

```python
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
for i,model_name in enumerate(models):
    model = models[model_name]
    scores = **cross_val_score**(model, X, y, cv = cv, 
                                                        n_jobs = -1, 
                            scoring = cv_scoring)
```

The mean estimated accuracy score after k-fold cross-validation for all proposed machine-learning models was $100\% \pm 0$. 

### Training Models

The next step was to plot a confusion matrix for each model and check the accuracy with the 20% of cross-validation data. 

```python
def runmodel(model,X_train,X_cv,y_train,y_cv,model_name):
    model.fit(X_train, y_train)
    preds = model.predict(X_cv)
    printresults(model,X_train,y_train,y_cv,preds,model_name)

models = {"SVC": SVC(),
    "Gaussian NB": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=18)}

for model_name in models:
    print('=='*30)
    runmodel(models[model_name],X_train,X_cv,y_train,y_cv,model_name)
```

![Figure 2.](https://raw.githubusercontent.com/LexC/ML_Disease_Prediction/main/outputs/cm_cv_svc.png)

Figure 2.

```python
============================================================
Accuracy on train data by SV : 100.0
Accuracy on test data by SV : 100.0
```

![Figure 3](https://raw.githubusercontent.com/LexC/ML_Disease_Prediction/main/outputs/cm_cv_nb.png)

Figure 3

```python
============================================================
Accuracy on train data by NB : 100.0
Accuracy on test data by NB : 100.0
```

![Figure 4](https://raw.githubusercontent.com/LexC/ML_Disease_Prediction/main/outputs/cm_cv_rf.png)

Figure 4

```python
============================================================
Accuracy on train data by RF : 100.0
Accuracy on test data by RF : 100.0
```

Figures 2, 3 and 4 and the accuracy of 100% show that the models are performing well on the unseen cross-validation data. Then, the last step is to combine the models and evaluate the accuracy in the testing dataset.

### **Fitting the model and validating on the Test dataset**

The code below shows how the models were combined. 

```python
from scipy.stats import mode

final_models = {    "svm": SVC(),
    "nb" : GaussianNB(),
    "rf" : RandomForestClassifier(random_state=18)  }

preds = {}
for m in final_models.keys():
    final_models[m].fit(X_train,y_train)
    preds[m] = final_models[m].predict(X_test)

final_preds = [mode([i,j,k]).mode for i,j,k in zip(preds['svm'], preds['nb'], preds['rf'])]
```

With the combined model, the resulting confusion matrix (figure 5) shows that all the data points were accurately classified.

```python
Accuracy on Test dataset by the combined model: 100.0
```

![Figure 5](https://raw.githubusercontent.com/LexC/ML_Disease_Prediction/main/outputs/cm_test_combined.png)

Figure 5