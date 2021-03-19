# kass-p2

## Requirements
- Apache Spark
- Tensorflow, Keras

## Linear Methods:
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

## Neural Network Approaches
### Keras Models with Spark
### Keras Models without Spark
## Contributions
Please see [CONTRIBUTORS](https://github.com/dsp-uga/kass-p2/blob/main/CONTRIBUTORS.md) file for more details.
## Authors 
- [Divya Yadava](https://github.com/YDivyaKrishna)
- [Kadir Bice](https://github.com/kbice)
- [Yogesh Chaudhari](https://github.com/yogeshchaudhari)


## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/dsp-uga/kass-p2/blob/main/LICENSE) file for the details.

## References: 
- https://spark.apache.org/docs/latest/ml-classification-regression.html
-  https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-classification-with-python-code/
- https://docs.databricks.com/_static/notebooks/deep-learning/deep-learning-transfer-learning-keras.html
- https://medium.com/linagora-engineering/making-image-classification-simple-with-spark-deep-learning-f654a8b876b8
- https://smurching.github.io/spark-deep-learning/site/api/python/sparkdl.html#sparkdl.readImages
