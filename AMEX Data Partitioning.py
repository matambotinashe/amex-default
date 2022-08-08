# Databricks notebook source
# MAGIC %md
# MAGIC # **Amex Data Partitioning**
# MAGIC - In this notebook partition the data for predict credit default for American Express.
# MAGIC - This is for the competiton on kaggle link https://www.kaggle.com/competitions/amex-default-prediction.
# MAGIC - Instead of using kaggle notebooks we have opted for databricks as the data is very large and could better handled with parrallel processing.
# MAGIC - This aim to accelarate our trial and error exercise in feature engineering before we are satisfied with the approach and apply it to all our data.
# MAGIC - It will also help us on making sure that we do not uncessarily use large resource when we still experimenting.
# MAGIC 
# MAGIC 
# MAGIC ## __Import Data__
# MAGIC - Now will import just the train data, test data.
# MAGIC - The is stored on Azure blod container.
# MAGIC - The path to the data are in a separate notebook 'path_config' and will be ignored by git.

# COMMAND ----------

# MAGIC %run ./path_config

# COMMAND ----------

train_data = spark.read.option('header', True).csv(train_data_path, inferSchema = True)
train_data.display()

# COMMAND ----------

test_data = spark.read.option('header', True).csv(test_data_path, inferSchema = True)
test_data.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Save Data**
# MAGIC - Now will save the data
# MAGIC - We are not applying an rules on partition created.
# MAGIC - This mean the partitions are created when we write our dataframe back to our Azure blod container and do not use coalesce.

# COMMAND ----------

train_data.write.mode('overwrite').option('header', True).csv(train_data_partition_out)

# COMMAND ----------

test_data.write.mode('overwrite').option('header', True).csv(test_data_partition_out)
