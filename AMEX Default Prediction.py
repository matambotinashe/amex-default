# Databricks notebook source
# MAGIC %md
# MAGIC # **AMEX Default Prediction**
# MAGIC - In this notebook will be building a model to predict credit default for American Express.
# MAGIC - This is for the kaggle competiton on https://www.kaggle.com/competitions/amex-default-prediction.
# MAGIC 
# MAGIC ## **Libraries**
# MAGIC - Now will import libraries that will need in this notebook.
# MAGIC - Here will be using Logistic Regression algorithms select in the notebook **AMEX Model Selection**. .

# COMMAND ----------

import mlflow
import joblib
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

from imblearn.over_sampling import SMOTE

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Import Data**
# MAGIC - The final train features data produced by notebook **Amex Feature Engineering** with be used in this notebook.

# COMMAND ----------

features_data = pd.read_csv('/dbfs/FileStore/amex/train_features.csv')
features_data.head()

# COMMAND ----------

features_data.shape

# COMMAND ----------

# MAGIC %md
# MAGIC __Comments__
# MAGIC - Now will just make sure our data does not have null values.

# COMMAND ----------

features_data.describe()

# COMMAND ----------

sum(features_data.isnull().sum() > 0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Datasets**
# MAGIC - First will min max scaler our features so that they can values between 0 and 1.
# MAGIC - We will split our data into train, test and validation datasets on 80:10:10 ratio.
# MAGIC - Due to the class inbalance in our target columns will over sample our train dataset.
# MAGIC - The competition provided test dataset will be referred to as the submission dataset to avoid confusion from here onwards.

# COMMAND ----------

scaler = MinMaxScaler()

train_features = features_data.drop(['customer_id', 'target'], axis = 1).columns

y = np.array(features_data['target'].astype(int))
X = np.array(features_data[train_features])

X = scaler.fit(X).transform(X)

# COMMAND ----------

# MAGIC %md
# MAGIC __Comments__
# MAGIC - Now will split our dataset.
# MAGIC - We will first create a test set which is 20% of the data then further equlal split into test and valid sets.

# COMMAND ----------

random_seed = 44
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_seed)
X_valid, X_test, y_valid , y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state = random_seed)

# COMMAND ----------

X_train.shape, X_test.shape, X_valid.shape

# COMMAND ----------

# MAGIC %md
# MAGIC __Comments__
# MAGIC - Now over sample our train data sets to address class imbalance.

# COMMAND ----------

y_train.mean()

# COMMAND ----------

X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train, y_train)
y_train_resampled.mean(), X_train_resampled.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Logistic Regression**
# MAGIC - Now will train a Logistic Regression model on our data.
# MAGIC - We are going to use the hyperparameter we determined as best in the AMEX Model Selection notebooks.

# COMMAND ----------

C_ = 0.36257120538682047 #0.07727850835507746
l1_ratio = 0.8994364898813669 #0.6627998959125167
max_iter = 900 #2_300
solver = 'newton-cg' #'lbfgs'
penalty = 'none'

# COMMAND ----------

# MAGIC %md
# MAGIC ### **1. Model Training**
# MAGIC - Now will train our model

# COMMAND ----------

mlflow.autolog()

# COMMAND ----------

model = LogisticRegression(C = C_, solver = solver, max_iter = max_iter, l1_ratio = l1_ratio, verbose = 1)
model.fit(X_train_resampled, y_train_resampled)

# COMMAND ----------

joblib.dump(model, '/dbfs/FileStore/amex/performance/trial3/model.pkl')

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Perfomance**
# MAGIC - Now will look at the performance of our model
# MAGIC - Since we are using MLflow auto log, the following are already created for us on the training data
# MAGIC   1. Confusion Matrix
# MAGIC   2. ROC curve
# MAGIC   3. Precision-recall curve
# MAGIC - We display these in our notebook and recreate them for the test data.
# MAGIC - We are loading the model which was saved by mlflow.

# COMMAND ----------

model = joblib.load('/dbfs/FileStore/amex/performance/trial3/model.pkl')

# COMMAND ----------

def plot_perfomance(image_path, height = 700, width = 1050, dpi = 72):
  perfomanc_graph = mpimg.imread(image_path)
  
  fig = plt.figure(figsize = (width/dpi ,height/dpi))
  ax = fig.add_subplot(111)
  ax.set_axis_off()
  ax.imshow(perfomanc_graph, interpolation = 'none')
  plt.show()

# COMMAND ----------

plot_perfomance('/dbfs/FileStore/amex/performance/trial3/training_confusion_matrix.png', dpi = 128)

# COMMAND ----------

plot_perfomance('/dbfs/FileStore/amex/performance/trial3/training_roc_curve.png')

# COMMAND ----------

plot_perfomance('/dbfs/FileStore/amex/performance/trial3/training_precision_recall_curve.png')

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Test Data Performance**
# MAGIC - Now will look at the test performance
# MAGIC - We will look at the following
# MAGIC   1. Accuracy,
# MAGIC   2. Confusion Matrix
# MAGIC   3. Precision-Recall curve
# MAGIC   4. ROC curve

# COMMAND ----------

def test_classification_report(model, X, y):
  y_predicted = model.predict(X)
  return print(classification_report(y,y_predicted))

# COMMAND ----------

model.score(X_test, y_test)

# COMMAND ----------

test_classification_report(model, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Important Features**
# MAGIC - We will look at the important features for our model

# COMMAND ----------

def model_import_features(model, maximum_features):
  smf = SelectFromModel(model, threshold = None, prefit = True, max_features = maximum_features)
  return smf.get_support()

# COMMAND ----------

feature_idx = model_import_features(model, 150)
feature_name = train_features[feature_idx]
feature_name

# COMMAND ----------

len(feature_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **2. Model training**
# MAGIC - Now will train the model on the top 110 most important features

# COMMAND ----------

def feature_impt_model(feature_index, X = X_train_resampled, y = y_train_resampled):
  """ This function take in a list of boolean value indicating whic features are importance, filter the X_train data based on them and trains a logistic model just on the import features.
  """
  X_important = np.transpose(np.transpose(X.copy())[feature_index])
  
  model_trial = LogisticRegression(C = C_, solver = solver, max_iter = max_iter, l1_ratio = l1_ratio, verbose = 1)
  model_trial.fit(X_important, y)
  
  return model_trial

# COMMAND ----------

def feature_impt_test(model, feature_index, X_ = X_test, y_ = y_test):
  """ This function test the performance on the model built on important features on the test data. """
  X_test_importanct = np.transpose(np.transpose(X_.copy())[feature_index])
  
  accuracy = model.score(X_test_importanct, y_)
  y_predicted = model.predict(X_test_importanct)
  
  return print(f'The model accuracy on the test data is {accuracy:.2%} and below is the classification report\n {classification_report(y_,y_predicted)}')

# COMMAND ----------

model2 = feature_impt_model(feature_idx)

# COMMAND ----------

feature_impt_test(model2, feature_idx)

# COMMAND ----------

# MAGIC %md
# MAGIC __Comments__
# MAGIC - The model did not perform significant better when trained on the 93 most important features.
# MAGIC - The chart log by mlflow show similar performance to the one we intial had.
# MAGIC 
# MAGIC 
# MAGIC ### **3. Model training**
# MAGIC - Will further reduce the number of important features to 75 and retrain our model still using the same hyperparameters.

# COMMAND ----------

feature_idx = model_import_features(model, 110)

# COMMAND ----------

model3 = feature_impt_model(feature_idx, X_train_resampled, y_train_resampled)
feature_impt_test(model3, feature_idx)

# COMMAND ----------

# MAGIC %md
# MAGIC __Comments__
# MAGIC - When we reduced the number of important feature down to 75 features, the accuracy of the model drop for to 86.73% from 87.04%.

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Validation**
# MAGIC - Now will see how our model2 perform on the validation data.

# COMMAND ----------

feature_idx = model_import_features(model, 110)
feature_impt_test(model3, feature_idx, X_valid, y_valid)

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Conclusion**
# MAGIC - Our model performed well on the validation data
# MAGIC - For our submission will consider the model with 93 features.
# MAGIC - Will save this model and the feature names that it uses.

# COMMAND ----------

feature_name = train_features[feature_idx]
with open('/dbfs/FileStore/amex/performance/trial3/feature_name.txt', 'w+') as f:
  for items in feature_name:
        f.write('%s\n' %items)
     
  print("File written successfully")
f.close()

# COMMAND ----------

joblib.dump(model3, '/dbfs/FileStore/amex/performance/trial3/model3.pkl')
