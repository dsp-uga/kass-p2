#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 15:21:13 2021

@author: kadir
"""
import sys
from pyspark.sql import *
from pyspark.sql.functions import col

from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, NaiveBayes, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler, MinMaxScaler

spark = SparkSession.builder.appName('deneme1').getOrCreate()

input_xtrain_file = sys.argv[1] #"/home/kadir/Documents/Spyder/DSP_Spring21/project2/X_small_train.csv" # 
input_xtest_file = sys.argv[2] #"/home/kadir/Documents/Spyder/DSP_Spring21/project2/X_small_test.csv" #
input_linear_method = sys.argv[3] # 'linearRegression'
output_file = sys.argv[4] # "/home/kadir/Documents/Spyder/DSP_Spring21/project2/output"

df_training = spark.read.option("header", "true").csv(input_xtrain_file)
df_test = spark.read.option("header", "true").csv(input_xtest_file)

# df_training.show(10, truncate = 3)
# df_training.select([df_training.columns[i] for i in range(9,176)])

# select features and convert to numeric
def gather_features(X_dataframe, isTestSet = False):
    processed = X_dataframe.select([X_dataframe.columns[i] for i in range(9,len(X_dataframe.columns))])
    processed = processed.select([col(c).cast("float") for c in processed.columns])
    
    # get column names
    cols = processed.columns
    
    # remove labels(sex), skip if test set
    if isTestSet == False:
        processed = processed.withColumnRenamed("Sex (subj)", "Sex")
        cols = processed.columns[:-1]
    
    # assemble features
    assembler = VectorAssembler(inputCols = cols, outputCol="features")
    processed = assembler.transform(processed)
    # training.select("features").show(10)
    
    # scale feature vector
    scaler = MinMaxScaler(inputCol = "features", outputCol = "Scaled_features")
    processed = scaler.fit(processed).transform(processed)
    # processed = processed.select("Scaled_features","Sex")
    # training.select("features","Scaled_features").show(5)
    
    return processed

# train and implement classifier
def linear_classifier_run(df_training, df_test, whichModel, isSmallSet = False):
    # gather train and test sets, if small set include Sex for accuracy testing
    train = gather_features(df_training).select("Scaled_features", "Sex")
    if isSmallSet == True:
        test = gather_features(df_test).select("Scaled_features", "Sex")
    else:
        test = gather_features(df_test, isTestSet = True).select("Scaled_features")

    # select classifier
    if whichModel == 'logisticRegression':    
        classifier = LogisticRegression(labelCol="Sex", featuresCol="Scaled_features", maxIter = 10)
    elif whichModel == 'onevsall':
        lr = LogisticRegression(labelCol="Sex", featuresCol="Scaled_features", maxIter=10)
        classifier = OneVsRest(classifier=lr, labelCol="Sex", featuresCol="Scaled_features")
    elif whichModel == 'decisionTree':
        classifier = DecisionTreeClassifier(labelCol="Sex", featuresCol="Scaled_features", maxDepth = 3)
    elif whichModel == 'randomForest':
        classifier = DecisionTreeClassifier(labelCol="Sex", featuresCol="Scaled_features")
    elif whichModel == 'gbt':
        classifier = GBTClassifier(labelCol="Sex", featuresCol="Scaled_features", maxIter = 10)
    elif whichModel == 'nb':
        classifier = NaiveBayes(labelCol="Sex", featuresCol="Scaled_features", smoothing=1.0, modelType="multinomial")
    else: 
        raise NameError("Model must be one of the following: logisticRegression, onevsall, decisionTree, randomForest, gbt or nb")
        
    # train the model with selected classifier
    model = classifier.fit(train)
        
    # predict test set
    print('Predicting with ', input_linear_method)
    predict_test = model.transform(test)
    # write to a text file
    predict_test.select('prediction').rdd.map(lambda x : str(int(x[0]))).saveAsTextFile(output_file)
    print('Output has been written to txt file')
    
    # test accuracy if small set
    if isSmallSet == True:
        results = predict_test.select("Sex","prediction").withColumn('Success', (predict_test['Sex'] == predict_test['prediction']))
        print('Accuracy of', whichModel, '= ', results.select("Success").where("Success == true").count() / results.count())

# run selected classifier
linear_classifier_run(df_training, df_test, whichModel = input_linear_method)

# training.show(10, truncate=3)
# training.dtypes

# training.withColumn("show", joindf["show"].cast(DoubleType()))
