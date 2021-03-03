# kass-p2

Linear Methods:
- Implemented using SparkMLlib with modules: pyspark.ml and pyspark.sql

- Alternative classifiers included (argument name): 
---Logistic Regression ('logisticRegression')
---One-vs-All ('onevsall')
---Decision Tree ('decisionTree')
---Random Forest ('randomForest')
---Gradient-Boosted Trees ('gbt')
---Naive Bayes ('nb')

- Input arguments: <x_train_file> <x_test_file> <classifier_selection> <output_directory>
---x_train_file: in csv format, has sex information in last column
---x_test_file: in csv format, does not have sex information in last column
---output_directory: should not be an existing directory

References: 
https://spark.apache.org/docs/latest/ml-classification-regression.html
