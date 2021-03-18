#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 21:41:06 2021

@author: kadir
"""
# Import libraries
from pyspark.sql import SparkSession
import pandas as pd
from PIL import Image
import numpy as np
import io

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql import functions as F
from pyspark.sql.functions import lit

# Creating SparkSession
spark = (SparkSession
            .builder
            .getOrCreate()
)

############ FEATURE EXTRACTION ROUTINE
# ResNet can be changed to VGG16, InceptionV3
model = VGG16(include_top=False)
bc_model_weights = spark.sparkContext.broadcast(model.get_weights())

def model_fn():
  """
  Returns a ResNet50 model with top layer removed and broadcasted pretrained weights.
  """
  model = VGG16(weights='imagenet', include_top=False) #ResNet50(weights='none', include_top=False)
  # model.set_weights(bc_model_weights.value)
  return model

def preprocess(content):
  """
  Preprocesses raw image bytes for prediction.
  """
  img = Image.open(io.BytesIO(content)).resize([224, 224])
  arr = img_to_array(img)
  return preprocess_input(arr)

def featurize_series(model, content_series):
  """
  Featurize a pd.Series of raw images using the input model.
  :return: a pd.Series of image features
  """
  input = np.stack(content_series.map(preprocess))
  preds = model.predict(input)

  output = [p.flatten() for p in preds]
  return pd.Series(output)

@pandas_udf('array<double>', PandasUDFType.SCALAR_ITER)
def featurize_udf(content_series_iter):
  '''
  This method is a Scalar Iterator pandas UDF wrapping our featurization function.
  The decorator specifies that this returns a Spark DataFrame column of type ArrayType(FloatType).
  
  :param content_series_iter: This argument is an iterator over batches of data, where each batch
                              is a pandas Series of image data.
  '''
  model = model_fn()
  for content_series in content_series_iter:
    yield featurize_series(model, content_series)

# Pandas UDFs on large records (e.g., very large images) can run into Out Of Memory (OOM) errors.
# If you hit such errors in the cell below, try reducing the Arrow batch size via `maxRecordsPerBatch`.
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")

############ END OF FEATURE EXTRACTION ROUTINE

############ COLLECT AND FORMAT DATA
train_1_images = spark.read.format("binaryFile") \
  .option("pathGlobFilter", "*.jpg") \
  .option("recursiveFileLookup", "true") \
  .load("/user/dsp_kass/data/faces_small/1").withColumn("label", lit(1))

train_0_images = spark.read.format("binaryFile") \
  .option("pathGlobFilter", "*.jpg") \
  .option("recursiveFileLookup", "true") \
  .load("/user/dsp_kass/data/faces_small/0").withColumn("label", lit(0))

test_1_images = spark.read.format("binaryFile") \
  .option("pathGlobFilter", "*.jpg") \
  .option("recursiveFileLookup", "true") \
  .load("/user/dsp_kass/data/faces_small/test/1").withColumn("label", lit(1))

test_0_images = spark.read.format("binaryFile") \
  .option("pathGlobFilter", "*.jpg") \
  .option("recursiveFileLookup", "true") \
  .load("/user/dsp_kass/data/faces_small/test/0").withColumn("label", lit(0))

# dataframe for training a classification model
train_df = train_1_images.unionAll(train_0_images)

# dataframe for testing the classification model
test_df = test_1_images.unionAll(test_0_images)

# Vectorize array<double> 
to_vector_udf = F.udf(lambda x: Vectors.dense(x), VectorUDT())

# Featurize train and test sets
train_features_df = train_df.select(col("path"), featurize_udf("content").alias("features"), col('label')).select(col('path'), to_vector_udf("features").alias('features_vec'), col('label'))
test_features_df = test_df.select(col("path"), featurize_udf("content").alias("features"), col('label')).select(col('path'), to_vector_udf("features").alias('features_vec'), col('label'))

# train_features_df.dtypes

# Train classification model (we can use other alternatives here)
classifier = LogisticRegression(labelCol="label", featuresCol="features_vec", maxIter = 10)
model = classifier.fit(train_features_df)

# Predict test set
predict_test = model.transform(test_features_df)
# Show results
# predict_test.select('label', 'prediction').show(5)
# Write to a text file
predict_test.repartition(1).select('prediction').rdd.map(lambda x : str(int(x[0]))).saveAsTextFile("/user/dsp_kass/output_udf")

# predict_test.repartition(1).select('label','prediction').rdd.map(lambda x : (str(int(x[0])), str(int(x[1])))).saveAsTextFile("/user/dsp_kass/output_udf")


