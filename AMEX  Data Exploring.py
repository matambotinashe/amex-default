# Databricks notebook source
# MAGIC %md
# MAGIC # __Amex EDA__
# MAGIC - In this notebook will be exploratory data analysis(EDA) on the data for predict credit default for American Express.
# MAGIC - This is for the kaggle competiton on https://www.kaggle.com/competitions/amex-default-prediction.
# MAGIC - Instead of using kaggle notebooks we have opted for databricks as the data is very large and databricks makes it easy to process it.
# MAGIC 
# MAGIC ## **Libraries**
# MAGIC - Now will import libraries that will need for this exercise.

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql.functions import *
from pyspark.sql.window import Window

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

# COMMAND ----------

# MAGIC %md
# MAGIC ## __Import Data__
# MAGIC - Now will import just the train data and labels to explore.
# MAGIC - The is stored on Azure blod container.
# MAGIC - The path to the data are in a separate notebook 'path_config' and will be ignored by git.

# COMMAND ----------

# MAGIC %run ./path_config

# COMMAND ----------

temp_path = train_data_partition_out + 'part-0000*.csv'

train_data = spark.read.option('header', True).csv(temp_path, inferSchema = True)
train_data.display()

# COMMAND ----------

# MAGIC %md
# MAGIC __Comments__
# MAGIC - We cached both our trainning dataframe as they are very big and process performed on them will be very slow.

# COMMAND ----------

train_labels = spark.read.option('header', True).csv(train_labels_path, inferSchema = True)
train_labels.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Data Exploring and Cleaning**
# MAGIC - Now will expore and clean our data.
# MAGIC - Will be focusing on the train_data and will replicate process on the test_data.

# COMMAND ----------

train_data.printSchema()

# COMMAND ----------

# Get the shape of the train data.
train_data.count(), len(train_data.columns)

# COMMAND ----------

train_data.describe().display()

# COMMAND ----------

train_labels.describe().display()

# COMMAND ----------

train_labels.select('target').display()

# COMMAND ----------

# MAGIC %md
# MAGIC __Comments__
# MAGIC - We have class inbalance for the target variable and we will address this on modelling.
# MAGIC - Now will check if the unique number of customers in our train data correspond to those in the train labels data.
# MAGIC - Will also make sure we do not have null values in the customer_ID column for our train_data.

# COMMAND ----------

train_data.select('customer_ID').distinct().count(), train_labels.select('customer_ID').distinct().count()

# COMMAND ----------

train_data.filter( col('customer_ID').isNull()).display()

# COMMAND ----------

# MAGIC %md
# MAGIC __Comments__
# MAGIC - We do not have case of null values in the customer_ID column of the train_data.
# MAGIC - The number of unique customers we have in the train labels correspond to the one we have in the train data.
# MAGIC - This then implies that we have multiple entiries of data for for customers in the train data.
# MAGIC - Now will investigate the difference in these entries.
# MAGIC - As a start will look at the S_2 which is a date column, for possible of the entries belong to different dates.

# COMMAND ----------

train_data.filter( col('customer_ID') == '0000099d6bd597052cdcda90ffabf56573fe9d7c79be5fbac11a8ed792feb62a').display()

# COMMAND ----------

s_2_duplicates = train_data.groupby(['customer_ID', 'S_2']).agg( count('customer_ID').alias('same_date_entries'))
s_2_duplicates.filter( col('same_date_entries') > 1 ).display()

# COMMAND ----------

# MAGIC %md
# MAGIC __Comments__
# MAGIC - The different entries for the same customer_ID in our train data are due to different dates found in the column 'S_2'.
# MAGIC - Now will check if the dates found in column 'S_2' are the same for all our customers.

# COMMAND ----------

train_data.groupby('S_2').agg(count('customer_ID').alias('records_per_date')).sort(col('S_2').asc()).display()

# COMMAND ----------

# MAGIC %md
# MAGIC __Comments__
# MAGIC - We do not have the same dates for our customers as it looks like the data is just from 1 March 2017 to 31 March 2018.
# MAGIC - This implies that the dates we have are somehow customer driven be through settern attribute or activity.
# MAGIC 
# MAGIC ### S_2
# MAGIC - From this will further explore the column S_2
# MAGIC - Now will see if it is possible to have more that one record per month of one customer.

# COMMAND ----------

s2_data = train_data.select('customer_ID', 'S_2')\
  .withColumn('S_2_month', month('S_2'))\
  .withColumn('S_2_year', year('S_2')) # We add year to separate March 2017 from March 2018
s2_data.display()

# COMMAND ----------

monthly_records = s2_data.groupby(['customer_ID', 'S_2_month', 'S_2_year']).agg(count('S_2').alias('monthly_records'))
monthly_records.filter(col('monthly_records') > 1).display()

# COMMAND ----------

# MAGIC %md
# MAGIC __Comments__
# MAGIC - It looks like we have only one record per month for a customer.
# MAGIC - Now will investigate if every customer has a record for all the 13 months period we have in our data.
# MAGIC - Further will look into the number of days between consecutive days. 
# MAGIC - This will help us investigate wether the dates found in this column are customer driven or are due to system design or business rules.

# COMMAND ----------

months_occurance = monthly_records.groupby('customer_ID').agg(count('S_2_month').alias('month_occurance'))
months_occurance.display()

# COMMAND ----------

# MAGIC %md
# MAGIC __Comments__
# MAGIC - Not every customer appears 13 times in our data.
# MAGIC - Just look at the first 1000 rows, it appear that the number of months a customer appears is negatively skewed.
# MAGIC - Now will look at the numbers day between consecutive dates of each customer

# COMMAND ----------

date_window = Window.partitionBy('customer_ID').orderBy(col('S_2').asc())

s2_data = s2_data.withColumn('lag_date', lag('S_2', 1).over(date_window))\
  .withColumn('lag_days', datediff( col('S_2'), col('lag_date')))
s2_data.display()

# COMMAND ----------

s2_data.filter(col('customer_ID') == '000084e5023181993c2e1b665ac88dbb1ce9ef621ec5370150fc2f8bdca6202c').select('S_2_year', 'S_2_month', 'lag_days').display()

# COMMAND ----------

s2_data.filter(col('S_2_month') == 9).select('S_2_month','lag_days').display()

# COMMAND ----------

# MAGIC %md
# MAGIC __Comments__
# MAGIC - When we focus on rows of one customer, we noticed the variance in the number of days between consecutive days.
# MAGIC - When we focus on one month and look at the first 1000 rows, we noticed that the number of days between consecutive days is independant of the month.
# MAGIC - We can safely conclude that the dates found in column 'S_2' are customer driven.
# MAGIC 
# MAGIC #### **Conclusion**
# MAGIC - From this we can conclude that the number of months and days between consecutive days for our customer varies.
# MAGIC - This means that the dates are not due to system design or business rules.
# MAGIC - Will look at how these relate with other columns as we continue EDA.

# COMMAND ----------

train_data = train_data.withColumn('month', month('S_2'))\
  .withColumn('year', year('S_2'))\
  .withColumn('lag_date', lag('S_2', 1).over(date_window))\
  .withColumn('lag_days', datediff( col('S_2'), col('lag_date')))\
  .na.fill(value = 0, subset = ['lag_days'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Delinquency variables**
# MAGIC - Now we will look at the just the Delinquency variables( D_* columns)
# MAGIC - We also include the customer_ID, S_2 and other date associated columns.
# MAGIC - As we will be repeating the approach of filtering out groups of columns the other 4 variable types, hence we will create a function for this.

# COMMAND ----------

def variables_sub_df(df,variables_prefix,*args):
  """ 
    The function takes the variables prefix used in columns name of the variable type belong to our column and any other columns we also want to include.
    It will then return a dataframe that is a result of filtering df for the columns belong to the variables we intrested in and the other columns passed as args.
  """
  variables_col  = [col for col in train_data.columns if col.startswith(variables_prefix)]
  for col in args:
    variables_col.append(col)
  
  return  df.select(variables_col)

# COMMAND ----------

other_columns = ['customer_ID', 'S_2', 'lag_date', 'lag_days', 'month', 'year']

delinquency_columns_prefix = 'D_'

delinquency_data = variables_sub_df(train_data, delinquency_columns_prefix, *other_columns)
delinquency_data.display()

# COMMAND ----------

delinquency_data.describe().display()

# COMMAND ----------

# MAGIC %md
# MAGIC __Comments__
# MAGIC 
# MAGIC - The following columns have large propotion of null values
# MAGIC   - D42, D66, D73, D76, D87, D88, D108, D110, D111 and D134-138.
# MAGIC - For these columns will suggest droping them for this iteration.

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Categorical Columns**
# MAGIC - Now will explore the categorical columns 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66' and 'D_68'
# MAGIC - Will explore the possiblity of these categories remaining the same value for a customer regardless of the date.

# COMMAND ----------

delinquency_data.groupby('D_63').count().display()

# COMMAND ----------

delinquency_data.groupby(['customer_ID']).agg(countDistinct('D_63').alias('D_63_change'))\
  .filter( col('D_63_change') > 1).display()

# COMMAND ----------

delinquency_data.select('D_63', 'D_45').display()

# COMMAND ----------

delinquency_data.groupby('D_64').count().display()

# COMMAND ----------

delinquency_data.groupby(['customer_ID']).agg(countDistinct('D_64').alias('D_64_change'))\
  .filter( col('D_64_change') > 1).display()

# COMMAND ----------

delinquency_data.groupby('D_66').count().display()

# COMMAND ----------

delinquency_data.groupby(['customer_ID']).agg(countDistinct('D_66').alias('D_66_change'))\
  .filter( col('D_66_change') > 1).display()

# COMMAND ----------

delinquency_data.groupby('D_68').count().display()

# COMMAND ----------

delinquency_data.groupby(['customer_ID']).agg(countDistinct('D_68').alias('D_68_change'))\
  .filter( col('D_68_change') > 1).display()

# COMMAND ----------

delinquency_data.groupby('D_114').count().display()

# COMMAND ----------

delinquency_data.groupby(['customer_ID']).agg(countDistinct('D_114').alias('D_114_change'))\
  .filter( col('D_114_change') > 1).display()

# COMMAND ----------

delinquency_data.groupby('D_116').count().display()

# COMMAND ----------

delinquency_data.groupby(['customer_ID']).agg(countDistinct('D_116').alias('D_116_change'))\
  .filter( col('D_116_change') > 1).display()

# COMMAND ----------

delinquency_data.groupby('D_117').count().display()

# COMMAND ----------

delinquency_data.groupby(['customer_ID']).agg(countDistinct('D_117').alias('D_117_change'))\
  .filter( col('D_117_change') > 1).display()

# COMMAND ----------

delinquency_data.groupby('D_120').count().display()

# COMMAND ----------

delinquency_data.groupby(['customer_ID']).agg(countDistinct('D_120').alias('D_120_change'))\
  .filter( col('D_120_change') > 1).display()

# COMMAND ----------

delinquency_data.groupby('D_126').count().display()

# COMMAND ----------

delinquency_data.groupby(['customer_ID']).agg(countDistinct('D_126').alias('D_126_change'))\
  .filter( col('D_126_change') > 1).display()

# COMMAND ----------

# MAGIC %md
# MAGIC __Comments__
# MAGIC - After exploring our categorical columns, here are proposal of how to deal with null values,
# MAGIC   - D_64, D_117 and D_126 will replace with -1
# MAGIC   - D_68 will replace with 6
# MAGIC   - D_114, D_116 will replace with 1
# MAGIC   - D_120 will replace with 0
# MAGIC   - D_144 will random replace with both 0 and 1 while trying to maintain the proportion of rows with 0 and 1
# MAGIC - All these is just based on look and the mode category and revent proportions.
# MAGIC - Columns D_63 and D_64 will need to be one hot encoded.
# MAGIC - The values of the categorical columns change with date for customers.
# MAGIC - For the mean time the numerical columns null values won't be addressed.

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Spend Variables**
# MAGIC - Now will look into spend variables.

# COMMAND ----------

spend_columns_prefix = 'S_'

spend_data = variables_sub_df(train_data, spend_columns_prefix, other_columns[0])
display(spend_data)

# COMMAND ----------

spend_data.describe().display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Payment variables**
# MAGIC - Now will look into payment columns, will also include the date column S_2

# COMMAND ----------

payment_columns_prefix = 'P_'

payment_data =  variables_sub_df(train_data, payment_columns_prefix, *other_columns)
display(payment_data)

# COMMAND ----------

payment_data.describe().display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Balance variables**
# MAGIC - Now will look at the balance variables, as before will include S_2 column.

# COMMAND ----------

balance_columns_prefix = 'B_'

balance_data = variables_sub_df(train_data, balance_columns_prefix, *other_columns)
display(balance_data)

# COMMAND ----------

balance_data.describe().display()

# COMMAND ----------

# MAGIC %md
# MAGIC __Comments__
# MAGIC - They is a large propotion of null values in columns B_39 and B_42.
# MAGIC - For these columns will suggest droping them for this iteration.
# MAGIC - We will look at the categorical columns of the Balance variables which are 'B_30' and 'B_38'
# MAGIC - We first look into 'B_30'

# COMMAND ----------

balance_data.groupby('B_30').count().display()

# COMMAND ----------

balance_data.groupby(['customer_ID']).agg(countDistinct('B_30').alias('B_30_change'))\
  .filter( col('B_30_change') > 1).display()

# COMMAND ----------

# MAGIC %md
# MAGIC __Comments__
# MAGIC - For the same customer B_30 changes with date.
# MAGIC - Now will investigate the best approach to fill the null values in B_30.
# MAGIC - We will first look at other Balance Variables and we if they are any columns we can use to determine the best value to replace the nulls.
# MAGIC - First will look if this is anyhow related to the date column 'S_2' be earliest or latest date entry for each customer.
# MAGIC - Secondly will look at columns description of entries of those that have null's in B_30 and those that do not. 

# COMMAND ----------

balance_data.filter( col('B_30').isNull()).describe().display()

# COMMAND ----------

balance_data.filter( col('B_30').isNotNull()).describe().display()

# COMMAND ----------

# MAGIC %md
# MAGIC __Comments__
# MAGIC - After look at the description, we noticed that all 2016 rows with B_30 null have value 1 for B_31.
# MAGIC - Will now look and values usual B_30 takes when B_31 is 1.

# COMMAND ----------

balance_data.groupby('B_31').count().display()

# COMMAND ----------

# MAGIC %md
# MAGIC __Comments__
# MAGIC - Due to the imbalance in the column B_31, when can not use if to fill null values in B_30.
# MAGIC - For the null values will replace with 0 as it is the mode value of the column.
# MAGIC - Now will look at column B_38.

# COMMAND ----------

balance_data.groupby('B_38').count().display()

# COMMAND ----------

balance_data.groupby(['customer_ID']).agg(countDistinct('B_38').alias('B_38_change'))\
  .filter( col('B_38_change') > 1).display()

# COMMAND ----------

# MAGIC %md
# MAGIC __Comments__
# MAGIC - B_38 category also changes with date just like B_30.
# MAGIC - Rows that had null values in category B_30 also have null values in B_38.
# MAGIC - Will also replace null value in B_38 with mode value of 2.

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Conclusion**
# MAGIC - This concludes our data exploring, will be implimenting what we discovered here in our the **Amex Feature Engineering**.
