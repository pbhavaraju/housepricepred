#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('apt-get install openjdk-8-jdk-headless -qq > /dev/null')
get_ipython().system('pip install pyspark')
get_ipython().system('pip install -q findspark')


# In[2]:


from pyspark import SparkContext
from pyspark.sql.session import SparkSession
#from pyspark.streaming import StreamingContext
#import pyspark.sql.types as tp
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql import Row, Column, DataFrame
import sys
from pyspark.sql.functions import isnan, when, count, col
import matplotlib.pyplot as plt
import pandas
import seaborn as sns
from pyspark.mllib.stat import Statistics # for corelation matrix
from pyspark.sql.functions import year, month, current_date # for sale recency calculation
from pyspark.sql.functions import to_date  # for sale recency calculation
#from pyspark.sql.types import *
from typing import Iterable
from pyspark.ml.feature import Imputer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator


# In[3]:


sc = SparkContext.getOrCreate();
spark = SparkSession(sc)


# In[4]:


# read a csv file
my_data = spark.read.option('header','true').csv('train.csv',inferSchema=True)
test_data= spark.read.option('header','true').csv('test.csv',inferSchema=True)


# In[5]:


# see the default schema of the dataframe
my_data.printSchema()
my_data.show(5)


# In[6]:


test_data.printSchema()
test_data.show(5)


# we observe that 2 columns are intefered as String as againt Integer. these columns are "LotFrontage" and "MasVnrArea".
# 
# Lets convert these to Integer
# 

# In[7]:


from pyspark.sql.types import IntegerType
my_data = my_data.withColumn("LotFrontage", my_data["LotFrontage"].cast(IntegerType()))
my_data = my_data.withColumn("MasVnrArea", my_data["MasVnrArea"].cast(IntegerType()))
test_data = test_data.withColumn("LotFrontage", test_data["LotFrontage"].cast(IntegerType()))
test_data = test_data.withColumn("MasVnrArea", test_data["MasVnrArea"].cast(IntegerType()))


# In[8]:


my_data.printSchema()


# In[9]:


# check number of rows & colums of the Dataframe
# also check if any duplicate rows
data_row = my_data.count()
data_col = len(my_data.columns)
test_row = test_data.count()
test_col = len(test_data.columns)
# extracting number of distinct rows for train and test data
data_uniq_row = my_data.distinct().count()
test_uniq_row = test_data.distinct().count()
print(f'Dimension of the Train data is (row x col): {(data_row,data_col)}')
print(f'Distinct Number of Train data Rows are: {data_uniq_row}')
print(f'Dimension of the Test data is (row x col): {(test_row,test_col)}')
print(f'Distinct Number of Test data Rows are: {test_uniq_row}')


# **Check for Duplicate Rows, Null and missing data**
# 
# based on above results there are no duplicate rows as number of disctict rows matches with totas rows (1460 in Train data  and 1459 in test data)

# In[10]:


#for Train Data check if any NULL, None, NaN and missing data
from pyspark.sql.functions import isnan, when, count, col
my_data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in my_data.columns]).show()


# Lotfrontage has  259 entirs of "NA", we can impute using median.
# 
# MasVnrArea has 8 entres of "NA" there we can impute 0

# In[11]:


#for Test Data check if any NULL, None, NaN and missing data
from pyspark.sql.functions import isnan, when, count, col
test_data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in test_data.columns]).show()


# In[12]:


#Train data - check statistics analysis
my_data.describe().show(5)


# Lets analyze the data columns by going through data description and above statistics.
# 
# 1. "MoSold" and "YrSold" : Month of sold and Year of sold; we will use these columns to create "HosueAge_months", indicating age of house when sold
# 2. YearBuilt, YearRemodAdd: we will convert these year information to "Buit_Recency" and "Mod_recency". Indicating number months of built to sale and from Modification to sale
# 3. "MSSubClass", "OverallQual", "OverallCond" These 3 columns are labelled columns. we will use OHE on these
# 

# In[13]:


#store all column names that are categorical and count these
catCols = [x for (x, dataType) in my_data.dtypes if dataType =="string"]
#store all column names that are integer and count these
intCols = [x for (x, dataType) in my_data.dtypes if (dataType =="int") ]
print("categorial columns:")
print(catCols)
print("integer columns:")
print(intCols)
print()
print("count of categorial columns=",len(catCols))
print("count of integer columns=",len(intCols))


# **Correlation analysis on integer columns**

# In[14]:


dicts ={}
for x in intCols:
  a=my_data.stat.corr( x, 'SalePrice' )
  dicts[x]=a
  #print("Correlation of {} {}" .format (x,a))
print ("correlation values in descending")
a1_sorted_keys = sorted(dicts, key=dicts.get, reverse=True)
for r in a1_sorted_keys:
    print(r, dicts[r])


# we can drop colums which are having corelation between 0.1 to -0.1

# In[15]:


int_col_to_drop=[]
for k, v in dicts.items():
  if (v<0.1) & (v>-0.1):
    print (k)


# we can drop below columns
# [Id, MSSubClass, OverallCond, BsmtFinSF2,LowQualFinSF, BsmtHalfBath,3SsnPorch,PoolArea,MiscVal]
# MoSold and YrSold are Date columns and we can convert the to recency
# 

# In[16]:


#code to drop columns passed as banned_list
class ColumnDropper(Transformer):
    """
    A custom Transformer which drops all columns that have at least one of the
    words from the banned_list in the name.
    """

    def __init__(self, banned_list: Iterable[str]):
        super(ColumnDropper, self).__init__()
        self.banned_list = banned_list

    def _transform(self, df: DataFrame) -> DataFrame:
        df = df.drop(*[x for x in df.columns if any(y in x for y in self.banned_list)])
        return df


# In[17]:


#code to handle date columns
class DateHandler(Transformer):
    """
    A custom Transformer 
    """

    def __init__(self):
        super(DateHandler, self).__init__()
        #self.banned_list = banned_list

    def _transform(self, df: DataFrame) -> DataFrame:
        from pyspark.sql import functions as F
        df= df.withColumn("Current_year", year(F.current_date()))          .withColumn("Current_month", month(F.current_date()))
        df= df.withColumn("HosueAge_months",(df.Current_year - df.YrSold)*12                    +(df.Current_month - df.MoSold))            .withColumn("built_recency",(df.YrSold - df.YearBuilt)*12)            .withColumn("mod_recency",(df.YrSold - df.YearRemodAdd)*12)
        df=df.drop("Current_year","Current_month") #these columns added for temporary
        return df


# In[49]:


#define stage 0: Imputing "LotFrontage" column with median

catCols = [x for (x, dataType) in my_data.dtypes if dataType =="string"]
print(catCols)
#store all column names that are integer and count these
intCols = [x for (x, dataType) in my_data.dtypes if (dataType =="int") ]
                


# In[63]:


#define stage 0: Imputing "LotFrontage" column with median
imputer_1= Imputer(inputCol="LotFrontage", outputCol= "LotFrontage_i" ).            setStrategy("median")

#define stage 1: Imputing "MasVnrArea" column with "0"
I_value=0
imputer_2= Imputer(inputCol="MasVnrArea", outputCol= "MasVnrArea_i" ).setStrategy("median")

# define stage 2: handle date columns to create "Recency" infromation.
# it adds columns e.g. Age (in months) of the hosue from build date till sale date
# Sale yr and sale month info is converted sale date recency in months from current date
#below variable hold 3 columns names that are added as recency
#date_recency_col=["built_recency","mod_recency","HosueAge_months"] 
Date_Handler = DateHandler()


# define stage 3: convert categorial columns to labled 
catCols = [x for (x, dataType) in my_data.dtypes if dataType =="string"]
cat_labeled = [x+"_labled" for x in catCols]
string_indexer = StringIndexer(inputCols= catCols, outputCols= cat_labeled)


labeld_columns =["MSSubClass", "OverallQual", "OverallCond"]
labled_OHE= [x+"_encoded" for x in labeld_columns]
Labled_Encoder1 = OneHotEncoder(inputCols=labeld_columns, 
                               outputCols= labled_OHE)

cat_OHE = [x+"_encoded" for x in catCols]
Labled_Encoder2 = OneHotEncoder(inputCols=cat_labeled, 
                               outputCols= cat_OHE)

#--------------------Vector Assembler-------------------
col_imuted=["LotFrontage_i","MasVnrArea_i"]
date_recency_col=["built_recency","mod_recency","HosueAge_months"] 
col_int=['LotArea','BsmtFinSF1',          'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',          'GrLivArea', 'BsmtFullBath', 'FullBath',         'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 
         'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',\
         'ScreenPorch']

#define stage 6: create a vector of all the features required to train  model 
Vector_Assembler = VectorAssembler(
    inputCols= col_imuted + date_recency_col + col_int + labled_OHE + cat_OHE,
                          outputCol='features')
#----------------
rf = RandomForestRegressor(labelCol="SalePrice", featuresCol="features")    


# In[64]:


stages= [imputer_1, imputer_2, Date_Handler,string_indexer,Labled_Encoder1, Labled_Encoder2,Vector_Assembler,rf]
# setup the pipeline
pipeline = Pipeline().setStages(stages)
# fit the pipeline for the trainind data
model = pipeline.fit(my_data)
# transform the data
my_data1 = model.transform(my_data)

# view some of the columns generated
#my_data1.select('features', "SalePrice" 'prediction').show(5)
my_data1.show(5)
print(type(my_data1))


# In[65]:


my_data1.select('features', 'SalePrice', 'prediction')


# In[66]:


evaluator = RegressionEvaluator(
    labelCol="SalePrice", predictionCol="prediction", metricName="rmse")


# In[67]:


rmse = evaluator.evaluate(my_data1)

print("Root Mean Squared Error (RMSE) on  data = %g" % rmse)


# In[72]:


y_true = my_data1.select("SalePrice").toPandas()
y_pred = my_data1.select("prediction").toPandas()

import sklearn.metrics
r2_score = sklearn.metrics.r2_score(y_true, y_pred)
print('r2_score: {0}'.format(r2_score))

