# Databricks notebook source
# MAGIC %md
# MAGIC # **AMEX Model Selection**
# MAGIC - In this notebook will identify the best model and hyperparameter to predict credit default for American Express.
# MAGIC - This follows the after data exploring in the notebook **Amex Feature Engineering**.
# MAGIC - This is for the kaggle competiton on https://www.kaggle.com/competitions/amex-default-prediction.
# MAGIC 
# MAGIC ## **Libraries**
# MAGIC - Now will import libraries that will need in this notebook.
# MAGIC - Here will be using scikit-learn algorithms.

# COMMAND ----------

import mlflow
import pandas as pd
import numpy as np

import plotly.graph_objs as go
import matplotlib.pyplot as plt

import seaborn as sb

from sklearn.model_selection import cross_val_score,train_test_split,KFold,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import recall_score,roc_curve, roc_auc_score,f1_score,classification_report, confusion_matrix 
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.preprocessing import scale,Binarizer,MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from IPython.display import SVG,Image
from itertools import compress
from sklearn import tree

from imblearn.over_sampling import SMOTE

import joblib

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK, space_eval
from hyperopt.pyll import scope

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Import Data**
# MAGIC - The final train features data produced by notebook **Amex Feature Engineering** with be used in this notebooks.

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

y = np.array(features_data['target'].astype(int))
X = np.array(features_data.drop(['customer_id', 'target'], axis = 1))

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
# MAGIC ## **Hyperparameter Tuning and Model Selection**
# MAGIC - Since our default prediction is a binary classification, will try the following algorithmns
# MAGIC   1. decision tree
# MAGIC   2. logistic regression
# MAGIC   3. Random Forest
# MAGIC - We are going to use Hyperopt and MLflow for selection of the best algorithm for our data amoung the 3 options and also the best hyperparameters to use.
# MAGIC - Hyperopt will allow us to parallise to exercise of tuning and model selection.
# MAGIC - We are now going to enable MLflow autologging for this notebook.

# COMMAND ----------

mlflow.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Function to minimize**
# MAGIC - For each of the algorithm we going to try, we going to use **accuracy** as metric compare performance.
# MAGIC - We going to minimize the negative accuracy.
# MAGIC - We define a function to use for this.
# MAGIC - In order to make the process faster will  only use 25% of the train data in model selection and hyperparameters tuning.

# COMMAND ----------

def objective(params):
  X1, X2, y1, y2 = train_test_split(X_train_resampled, y_train_resampled, test_size = 0.25, random_state = 0)
  
  classifier_type = params['type']
  del params['type']
  if classifier_type == 'dtree':
    clf = DecisionTreeClassifier(**params)
  elif classifier_type == 'rf':
    clf = RandomForestClassifier(**params)
  elif classifier_type == 'logreg':
    clf = LogisticRegression(**params)
  elif classifier_type == 'logregnone':
    clf = LogisticRegression(**params)
  elif classifier_type == 'logregLl':
    clf = LogisticRegression(**params)
  elif classifier_type == 'logregSG':
    clf = LogisticRegression(**params)
  elif classifier_type == 'logregSGN':
    clf = LogisticRegression(**params)
  else:
    return 0
  accuracy = cross_val_score(clf, X2, y2, cv = 5).mean()
  #mlflow.log_metric("accuracy", accuracy)
  
  # Because fmin() tries to minimize the objective, this function must return the negative accuracy.
  return {'loss': -accuracy, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Search Space**
# MAGIC - Now will define the search space over hyperparameters for each of the model. 

# COMMAND ----------

# MAGIC %md
# MAGIC {
# MAGIC     'type': 'dtree',
# MAGIC     'criterion': hp.choice('criterion', ['gini', 'entropy']),
# MAGIC     'splitter': hp.choice('splitter', ['best', 'random']),
# MAGIC     'max_depth':hp.choice('max_depth', [None, hp.quniform('max_depth_int', 2, 300, 10)]),
# MAGIC     'min_samples_split': hp.quniform('min_samples_split', 0, 1, 0.001),
# MAGIC     'min_samples_leaf': hp.quniform('min_samples_leaf', 0, 1, 0.001),
# MAGIC     'max_features': hp.choice('max_features',
# MAGIC             [hp.choice('max_features_str',['auto', 'sqrt', 'log2']), hp.quniform('max_features_int', 2, 300,10)])
# MAGIC   },
# MAGIC   
# MAGIC   {
# MAGIC     'type': 'rf',
# MAGIC     'criterion': hp.choice('rf_criterion', ['gini', 'entropy']),
# MAGIC     'splitter': hp.choice('rf_splitter', ['best', 'random']),
# MAGIC     'max_depth':hp.choice('rf_max_depth', [None, hp.quniform('rf_max_depth_int', 2, 300, 10)]),
# MAGIC     'min_samples_split': hp.quniform('rf_min_samples_split', 0, 1, 0.001),
# MAGIC     'min_samples_leaf': hp.quniform('rf_min_samples_leaf', 0, 1, 0.001),
# MAGIC     'max_features': hp.choice('rf_max_features',
# MAGIC             [hp.choice('rf_max_features_str',['auto', 'sqrt', 'log2']), hp.quniform('rf_max_features_int', 2, 300, 10)])
# MAGIC     },

# COMMAND ----------

search_space = hp.choice('classifier_type', [
  {
    'type': 'dtree',
    'criterion': hp.choice('criterion', ['gini', 'entropy']),
    'splitter': hp.choice('splitter', ['best', 'random']),
    #'max_depth':hp.choice('max_depth', [None, hp.quniform('max_depth_int', 10, 300, 10)]),
    'min_samples_split': hp.quniform('min_samples_split', 0, 1, 0.001),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 0, 1, 0.001),
    'max_features': hp.choice('max_features',
            [hp.choice('max_features_str',['auto', 'sqrt', 'log2']), hp.quniform('max_features_int', 0, 1, 0.001)])
  },
  
  {
    'type': 'rf',
    'criterion': hp.choice('rf_criterion', ['gini', 'entropy']),
    #'max_depth':hp.choice('rf_max_depth', [None, hp.quniform('rf_max_depth_int', 10, 300, 10)]),
    'min_samples_split': hp.quniform('rf_min_samples_split', 0, 1, 0.001),
    'min_samples_leaf': hp.quniform('rf_min_samples_leaf', 0, 1, 0.001),
    'max_features': hp.choice('rf_max_features',
            [hp.choice('rf_max_features_str',['auto', 'sqrt', 'log2']), hp.quniform('rf_max_features_int',0, 1, 0.001)])
  },
  
  {
      'type': 'logreg',
      'penalty' : hp.choice('penaltyl2', ['l2']),
      'C': hp.lognormal('Cl2', 0, 1.0),
      'solver': hp.choice('solverl2', ['lbfgs', 'newton-cg', 'sag']),
      'max_iter': hp.quniform('max_iterl2', 100, 3000, 100)
  },
  {
      'type': 'logregnone',
      'penalty' : hp.choice('penalty', ['none']),
      'C': hp.lognormal('C', 0, 1.0),
      'solver': hp.choice('solver', ['lbfgs', 'newton-cg', 'sag']),
      'max_iter': hp.quniform('max_iter', 100, 3000, 100),
      'l1_ratio': hp.uniform('l1_ratio', 0.0, 1.0)
  },
  {
      'type': 'logregLl',
      'penalty' : hp.choice('penaltyLl', ['l1', 'l2']),
      'C': hp.lognormal('CL1', 0, 1.0),
      'solver': hp.choice('solverLl', ['liblinear']),
      'max_iter': hp.quniform('max_iterL1', 100, 3000, 100),
  },
  {
      'type': 'logregSG',
      'penalty' : hp.choice('penaltySG', ['l1', 'l2']),
      'C': hp.lognormal('CSG', 0, 1.0),
      'solver': hp.choice('solverSG', ['saga']),
      'max_iter': hp.quniform('max_iterSG', 100, 3000, 100),
  },
  {
      'type': 'logregSGN',
      'penalty' : hp.choice('penaltySGN', ['none', 'elasticnet']),
      'C': hp.lognormal('CSGN', 0, 1.0),
      'solver': hp.choice('solverSGN', ['saga']),
      'max_iter': hp.quniform('max_iterSGN', 100, 3000, 100),
      'l1_ratio': hp.uniform('l1_ratioSGN', 0.0, 1.0)
  }
])

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Search Algorithm**
# MAGIC - For our search algorithm will use hyperopt.tpe.suggest for that we can iterate to the best algorithm as we search.

# COMMAND ----------

algo=tpe.suggest
spark_trials = SparkTrials(parallelism = 4) 

# COMMAND ----------

with mlflow.start_run():
  best_result = fmin(
    fn=objective, 
    space=search_space,
    algo=algo,
    max_evals = 54,
    trials=spark_trials)

# COMMAND ----------

print(space_eval(search_space, best_result))

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Conclusion**
# MAGIC - From this we going to try a model with the best paramters show abouve.
