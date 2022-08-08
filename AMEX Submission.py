# Databricks notebook source
# MAGIC %md
# MAGIC # **AMEX Submission**s
# MAGIC - In this notebook will apply our model to predict default for American Express on the submission data.
# MAGIC - First will run our submission data through the notebook **AMEX Feature Engineering** by changing the path in cell 5 and 38.
# MAGIC - Will now apply prediction using the model created and logged in **AMEX Default Prediction**.
# MAGIC 
# MAGIC ## **Libraries**
# MAGIC - Now will import libraries that will need for applying prediction.

# COMMAND ----------

import mlflow
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# COMMAND ----------

# MAGIC %md
# MAGIC ## __Import Data__
# MAGIC - The submission data produced by running our submission data through the notebook **Amex Feature Engineering** with be used in this notebook.
# MAGIC - We will also import the list of features that our model will use.

# COMMAND ----------

# MAGIC %run ./path_config

# COMMAND ----------

submission_data = spark.read.option('header', True).csv(test_data_output, inferSchema = True).toPandas() #pd.read_csv('/dbfs/FileStore/amex/submission_features.csv')#
submission_data.head()

# COMMAND ----------

submission_data.shape

# COMMAND ----------

# MAGIC %md
# MAGIC __Comments__
# MAGIC - Now will make sure we do not have null values in our submission data.

# COMMAND ----------

sum(submission_data.isnull().sum() > 0)

# COMMAND ----------

# MAGIC %md
# MAGIC __Comments__
# MAGIC - Now will import the features for our model.

# COMMAND ----------

features_name = []

f = open('/dbfs/FileStore/amex/performance/trial2/feature_name.txt', 'r')
 
# display content of the file
for x in f.readlines():
  features_name.append(x.split('\n')[0])

# close the file
f.close()
features_name[:10]

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Datasets**
# MAGIC - First will min max scaler our features so that they can values between 0 and 1.

# COMMAND ----------

scaler = MinMaxScaler()

submission_features = submission_data[features_name]
customer_id = submission_data['customer_id']

X = np.array(submission_features)
X = scaler.fit(X).transform(X)
X.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Model Prediction**
# MAGIC - Now we going to import our model and predict default on the submission data.

# COMMAND ----------

model = joblib.load('/dbfs/FileStore/amex/performance/trial2/model2.pkl')

# COMMAND ----------

y_predict = model.predict(X)
y_predict[:10]

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Submission**
# MAGIC - Now will create our submission file and save it.

# COMMAND ----------

customer_id = list(customer_id)
prediction = list(y_predict)
submission = pd.DataFrame(list(zip(customer_id, prediction)), columns = ['customer_ID', 'prediction'])
submission.head()

# COMMAND ----------

submissionDF = spark.createDataFrame(submission)
submissionDF.display()

# COMMAND ----------

submissionDF.coalesce(1).write.mode('overwrite').option('header', True).csv(submission_path)

# COMMAND ----------

# MAGIC %md
# MAGIC __Comments__
# MAGIC - We verify the submission file.

# COMMAND ----------

submission_verification = spark.read.option('header', True).csv(submission_path, inferSchema = True)
submission_verification.display()
