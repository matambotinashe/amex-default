# Databricks notebook source
# MAGIC %md
# MAGIC # **Amex Feature Engineering**
# MAGIC - In this notebook will be generate features for predicting credit default for American Express.
# MAGIC - This follows the after data exploring in the notebook **Amex Data Exploring**.
# MAGIC - This is for the kaggle competiton on https://www.kaggle.com/competitions/amex-default-prediction.
# MAGIC - Instead of using kaggle notebooks we have opted for databricks as the data is very large and databricks makes it easy to process it.
# MAGIC 
# MAGIC ## **Libraries**
# MAGIC - Now will import libraries that will need for feature engineering.

# COMMAND ----------

import re
from pyspark.ml.feature import Imputer
from pyspark.sql.window import Window
from pyspark.sql.functions import col, max as max_, min as min_, mean, stddev_samp, row_number, lag, months_between, datediff, when

# COMMAND ----------

# MAGIC %md
# MAGIC ## __Import Data__
# MAGIC - Now will import our train data that is stored on Azure blod container.
# MAGIC - The path to the data are in a separate notebook 'path_config' and will be ignored by git.
# MAGIC - We are going to develop separate pipelines to apply the step take here for the test data.

# COMMAND ----------

# MAGIC %run ./path_config

# COMMAND ----------

train_data = spark.read.option('header', True).csv(train_data_path, inferSchema = True)
train_data.display()

# COMMAND ----------

# MAGIC %md
# MAGIC __Comments__
# MAGIC - Now will import the train labels.

# COMMAND ----------

train_labels = spark.read.option('header', True).csv(train_labels_path, inferSchema = True)
train_labels.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Data Cleaning**
# MAGIC - Now will create our features from the data.
# MAGIC - From our data exploring notebook will be doing the following steps
# MAGIC   - Will be dropping columns with large proportion of null values.
# MAGIC   - For categorical columns will replace null values with mode values as mentioned earlier.
# MAGIC   - Column D_63 and D_64 will be one hot encoded.

# COMMAND ----------

def data_cleaning(df, drop_columns, oneHotEncoderColumns):
  """ In this funtion will clean our train and test data by doing the following
    1. drop columns with a large proportion of na values.
    2. replace nulls with values determined during data exploring.
    3. One Hot encoder columns specified during the data exploring.
  """
  # Now will drop columns and replace null values
  df = df.drop(*drop_columns)\
    .na.fill(value = '-1', subset = ['D_64'])\
    .na.fill(value = -1, subset = ['D_64', 'D_117', 'D_126'])\
    .na.fill(value = 0, subset = ['D_68', 'D_144', 'D_120', 'B_30'])\
    .na.fill(value = 1, subset = ['D_114', 'D_116'])\
    .na.fill(value = 2, subset = ['B_38'])
  
  # Now will one hot encode our columns. Instead of using the OneHotEncoder transformation under the pyspark.ml.features
  # We have opted to do it manual as we do not want to deal with sparse vectors.
  # We want 
  for col_ in oneHotEncoderColumns:
    all_cats = df.select(col_).distinct().collect()
  
    # The number of columns we produce are one less that the number of categories in the original column for independant columns
    for i in range(len(all_cats)-1): 
      new_col = col_ + '_' + str(all_cats[i][0])
      df = df.withColumn(new_col, when( col( col_) == all_cats[i][0], 1).otherwise(0))
      
  
  return df.drop(*oneHotEncoderColumns)

# COMMAND ----------

drop_columns = ('B_39', 'B_42', 'D_42', 'D_66', 'D_73', 'D_76', 'D_87', 'D_88', 'D_108', 'D_110', 'D_111', 'D_134', 'D_135', 'D_136', 'D_137','D_138')
oneHotEncoderColumns = ('D_63', 'D_64')

train_data_cleaned = data_cleaning(train_data, drop_columns, oneHotEncoderColumns)

# COMMAND ----------

len(train_data_cleaned.columns)#, len(test_data_cleaned.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Features**
# MAGIC - Now we are going to create feature to be used in our prediction model.
# MAGIC - For output features, we going to group the data with customer_id and perform the following,
# MAGIC   - For numerical variable will calculate mean and standard deviation.
# MAGIC   - for categorical except those we one hot encoded will calculate the mode.
# MAGIC   - For the date column 'S_2' will calculate the number of days between the earliest and latest date.
# MAGIC   - Calculate the number of days between the earliest and latest date for each customer.
# MAGIC - For case in which we have more than one mode for a customer will take the first one to appear.

# COMMAND ----------

agg_columns = list(train_data_cleaned.columns)
agg_columns.remove('customer_ID')
agg_columns.remove('S_2')

mean_agg = {col_:'mean' for col_ in agg_columns}
col_agg = mean_agg.copy()

# COMMAND ----------

def data_agg(df, columns_aggegation):
  
  return df.groupby('customer_id').agg(columns_aggegation)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Mean Values**
# MAGIC - Now will compute the average value of our variables by grouping customer_ID.

# COMMAND ----------

train_features = data_agg(train_data_cleaned, col_agg)
train_features.display()

# COMMAND ----------

train_features.count(), len(train_features.columns)

# COMMAND ----------

train_features.describe().display()

# COMMAND ----------

# MAGIC %md
# MAGIC __Comments__
# MAGIC - The following columns are going to be dropped due to high numbers  of null values
# MAGIC   - 'avg(R26)', 'avg(D_138)', 'avg(D_66)', 'avg(D_88)', 'avg(D_137)', 'avg(R_9)', 'avg(D_111)', 'avg(D_73)', 'avg(D_142)', 'avg(D_132)', 'avg(D_134)', 'avg(D_49)', 'avg(D_136)', 'avg(D_42)', 'avg(B_29)', 'avg(D_110)', 'avg(D_87)', 'avg(D_135)', 'avg(D_106)'
# MAGIC   - avg('D_108'), avg('D_56'), 'avg(B_17)', avg(D_53), avg(D_50), avg(D_105), avg(D_82) is currently being drop but can later to reconsidered.

# COMMAND ----------

avg_drop_columns = ('avg(R26)', 'avg(D_138)', 'avg(D_66)', 'avg(D_88)', 'avg(D_137)', 'avg(R_9)', 'avg(D_111)', 'avg(D_73)', 'avg(D_142)', 'avg(D_132)', 'avg(D_134)', 
                    'avg(D_49)', 'avg(D_136)', 'avg(D_42)', 'avg(B_29)', 'avg(D_110)', 'avg(D_87)', 'avg(D_135)', 'avg(D_106)', 'avg(D_108)', 'avg(D_56)', 
                    'avg(B_17)', 'avg(D_53)', 'avg(D_50)', 'avg(D_105)', 'avg(D_82)')
train_features = train_features.drop(*avg_drop_columns)
len(train_features.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC __Comments__
# MAGIC - Will use Imputer to fill replace null values with mean value of each column.

# COMMAND ----------

input_cols = tuple(train_features.drop('customer_id').columns)
output_cols = tuple([re.sub('[^A-Za-z0-9]+', '_', col_) + 'impt' for col_ in input_cols])

imputer_ = Imputer(inputCols = input_cols, outputCols = output_cols).setStrategy("mean")

# After imputing, we drop the old columns.
train_features = imputer_.fit(train_features).transform(train_features).drop(*input_cols) 
train_features.describe().display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Standard Deviation**
# MAGIC - Now will compute the standard deviation for our variables grouped by customer_ID.
# MAGIC - Will apply the same steps we did for the mean values.
# MAGIC - Instead of drop columns with high number of null values post the grouping, we going to do it pre aggregation.

# COMMAND ----------

std_drop_columns = ('R26', 'D_138', 'D_66', 'D_88', 'D_137', 'R_9', 'D_111', 'D_73', 'D_142', 'D_132', 'D_134','D_49', 'D_136', 'D_42', 'B_29', 'D_110', 'D_87', 'D_135', 'D_106'
                    ,'D_108', 'D_56','B_17', 'D_53', 'D_50', 'D_105', 'D_82')
columns = train_data_cleaned.drop(*std_drop_columns).columns
columns.remove('customer_ID')
columns.remove('S_2')
std_agg = {col_: 'stddev_samp' for col_ in columns}

# COMMAND ----------

def null_value_replacement(df, strategy = 'mean'):
  input_cols = tuple(df.drop('customer_ID').columns)
  output_cols = tuple([re.sub('[^A-Za-z0-9]+', '_', col_) + 'impt' for col_ in input_cols])
  
  imputer_ = Imputer(inputCols = input_cols, outputCols = output_cols).setStrategy("mean")
  
  return  imputer_.fit(df).transform(df).drop(*input_cols)

# COMMAND ----------

train_features_std = data_agg(train_data_cleaned.drop(*std_drop_columns), std_agg)
train_features_std = null_value_replacement(train_features_std)
train_features_std.describe().display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categorical Columns
# MAGIC - Although we calculated mean and standard deviation on categorical columns, here will calculate the mode.
# MAGIC - For this will use window functions on each categorical column.
# MAGIC - Were we have more than one mode will just consider the first one to appear.
# MAGIC - Columns 'D_63', 'D_64' will be excluded as we one hot encoded them before.

# COMMAND ----------

def cat_col_mode(df, cat_col):
  """ 
    This function uses window functions to calculate the mode of a category column returning the mode and customer_ID.
    It will take the first appear mode incases we have more than one mode.
  """
  
  #We are going to partition on 'customer_ID' and consider the first row with the highest count. 
  windowSpec  = Window.partitionBy('customer_ID').orderBy(col('count').desc())
  
  #First group our data is grouped by 'customer_ID', our category column and count appears of the category. 
  mode_df = (
    train_data_cleaned.groupby('customer_ID', cat_col).count()
    .withColumn('row_num', row_number().over(windowSpec))
    .where(col('row_num') == 1)
    .select('customer_ID', cat_col)
    .withColumnRenamed(cat_col, f'mode_{cat_col}')
  )
  
  return mode_df

# COMMAND ----------

def join_df(df1, df2):
  """ This function outer joins dataframes on customer_ID column"""
  
  return df1.join(df2, on = 'customer_ID', how = 'outer')

# COMMAND ----------

cat_cols = ('B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_68')

cat_features = cat_col_mode(train_data_cleaned, 'B_30')

for cat_col in cat_cols:
  cat_features = join_df(cat_features, cat_col_mode(train_data_cleaned, cat_col))

cat_features.describe().display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dates
# MAGIC - Now will look at our date column 'S_2'
# MAGIC - Here are the varibles we going to compute from the date column
# MAGIC   - We will look at the number of day we have for each customer.
# MAGIC   - We compute the tenure of the customer that is the number of months between the earliest and latest date.
# MAGIC   - We will look at number of days between consecutive dates and from this will compute the mean and standard deviation of the number of days between consecutive days.
# MAGIC - For this will make use of window functions and group for the final data.

# COMMAND ----------

window = Window.partitionBy('customer_ID').orderBy(col('S_2').asc())

# For the earliest and latest date, we are using row between unbounded preceding and unbounded following 
date_window = Window.partitionBy('customer_ID').orderBy(col('S_2').asc()).rowsBetween(-9223372036854775808, 9223372036854775807)

dates_data = (
  train_data.select('customer_ID', 'S_2')
  .withColumn('earliest_date', min_('S_2').over(date_window)) 
  .withColumn('latest_date', max_('S_2').over(date_window))
  .withColumn('lag_date', lag('S_2', 1).over(window))
  .withColumn('months_tenure', months_between(col('latest_date'), col('earliest_date')))
  .withColumn('lag_days', datediff( col('S_2'), col('lag_date')))
)
dates_data.display()

# COMMAND ----------

dates_features = dates_data.groupby(['customer_ID', 'months_tenure'])\
  .agg( mean('lag_days').alias('avg_lag_days'), stddev_samp('lag_days').alias('std_lag_days'))\
  .na.fill(value = 0.0)
dates_features.display()

# COMMAND ----------

dates_features.describe().display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Final Features**
# MAGIC - Now will combine all our features into one dataframe including the train labels.
# MAGIC - We will then save out to our blod container.

# COMMAND ----------

final_train_features = train_features.join(train_features_std, on = 'customer_ID')\
  .join(cat_features, on = 'customer_ID')\
  .join(dates_features, on = 'customer_ID')\
  .join(train_labels, on = 'customer_ID')
final_train_features.describe().display()

# COMMAND ----------

# MAGIC %md
# MAGIC __Comments__
# MAGIC - Now we are ready to save our data to the blod contain and use in the modeling notebook.

# COMMAND ----------

final_train_features.coalesce(1).write.mode('overwrite').option('header', True).csv(train_data_output)
