import sys
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col
from pyspark.ml.classification import LogisticRegression
spark = SparkSession.builder.appName('image_classifier').getOrCreate()
x_train_file= sys.argv[1]
x_test_file = sys.argv[2]
output_path = "/home/dsp_kass/outputs/Logistic_Reg" #sys.argv[3]
train_df = spark.read.csv(x_train_file, header=True, inferSchema = True)
         
feature_columns = train_df.columns[9:-1]
assembler = VectorAssembler(inputCols= feature_columns, outputCol = "features")
train = assembler.transform(train_df)
             
      
test_df = spark.read.csv(x_test_file, header = True, inferSchema=True) 
test_assembler = VectorAssembler(inputCols=test_df.columns[9:], outputCol="features")
test = test_assembler.transform(test_df)
    
algo = LogisticRegression(featuresCol = "features", labelCol="Sex (subj)")
    
model = algo.fit(train)
    
predictions = model.transform(test) 
predictions.select('prediction').coalesce(1).rdd.map(lambda x: int(x[0])).saveAsTextFile(output_path)
print('predictions are stored in the output file')
