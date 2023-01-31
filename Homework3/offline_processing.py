#resource: https://cprosenjit.medium.com/9-classification-methods-from-spark-mllib-we-should-know-c41f555c0425
from pyspark.ml.feature import  VectorAssembler
from sklearn.model_selection import train_test_split
from pyspark.ml.feature import StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml import Pipeline
import pandas as pd
import pyspark
import os, sys


os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

data = pd.read_csv("offline.csv", delimiter=",")

#Split Offline data
train_df, test_df = train_test_split(data,test_size=0.2,random_state=42,stratify=data['Diabetes_binary'])

#create Spark Session
spark = pyspark.sql.SparkSession.builder.master("local[2]").appName("Domasna3").getOrCreate()

#create Spark DataFrame
train_df = spark.createDataFrame(train_df)
test_df = spark.createDataFrame(test_df)

#Print the columns
#train_df.printSchema()

#Print the first 5 rows
#train_df.show(5)

#Statistics on double features (all features in this case)
numeric_features = [t[0] for t in train_df.dtypes if t[1] == 'double']

#numeric_features = numeric_features.remove('Diabetes_binary')
#stats = train_df.select(numeric_features).describe().toPandas().transpose()
#print(stats)

#Create vector of features
vec_assembler = VectorAssembler(inputCols=numeric_features, outputCol="features")

#Using scaler to normalize the data as its features are in different range
stdScaler = StandardScaler(inputCol="features", \
                        outputCol="scaledFeatures", \
                        withStd=True, \
                        withMean=False)

#Creating Multinomial Logistic Regression model
lr1 = LogisticRegression(maxIter=20, \
                        regParam=0.1, \
                        elasticNetParam=0.1, \
                        featuresCol="scaledFeatures", \
                        family = "binomial", \
                        labelCol='Diabetes_binary')

lr2 = LogisticRegression(maxIter=100, \
                        regParam=0.3, \
                        elasticNetParam=0.1, \
                        featuresCol="scaledFeatures", \
                        family = "binomial", \
                        labelCol='Diabetes_binary')

lr3 = LogisticRegression(maxIter=70, \
                        regParam=0.5, \
                        elasticNetParam=0.1, \
                        featuresCol="scaledFeatures", \
                        family = "binomial", \
                        labelCol='Diabetes_binary')


#impurity = entoropy or gini
#supportedFeatureSubsetStrategies = ['auto', 'all', 'onethird', 'sqrt', 'log2']¶
#featureSubsetStrategy = Param(parent='undefined', name='featureSubsetStrategy',
# doc="The number of features to consider for splits at each tree node.
# Supported options: 'auto' (choose automatically for task: If numTrees == 1, set to 'all'.
# If numTrees > 1 (forest), set to 'sqrt' for classification and to 'onethird' for regression), 'all' (use all features),
# 'onethird' (use 1/3 of the features), 'sqrt' (use sqrt(number of features)), 'log2' (use log2(number of features)),
# 'n' (when n is in the range (0, 1.0], use n * number of features. When n is in the range (1, number of features),
# use n features). default = 'auto'")¶
rf1 = RandomForestClassifier(labelCol="Diabetes_binary", \
                            featuresCol="scaledFeatures", numTrees=5, maxDepth=2,\
                            impurity='gini',featureSubsetStrategy='onethird')

rf2 = RandomForestClassifier(labelCol="Diabetes_binary", \
                            featuresCol="scaledFeatures", numTrees=10, maxDepth=5,\
                            impurity='gini',featureSubsetStrategy='all')

rf3 = RandomForestClassifier(labelCol="Diabetes_binary", \
                            featuresCol="scaledFeatures", numTrees=10, maxDepth=5,\
                            impurity='entropy',featureSubsetStrategy='sqrt')


# specify layers for the neural network:
# input layer of size 22 (size of scaledFeatures), two intermediate of size 5 and 4
# and output of size 3 (classes)
layers1 = [22, 5, 4, 2]
mlp1 = MultilayerPerceptronClassifier(labelCol="Diabetes_binary",featuresCol="scaledFeatures",
                                      maxIter=50, layers=layers1, blockSize=128, seed=1234)

layers2 = [22, 4, 12, 4, 2]
mlp2 = MultilayerPerceptronClassifier(labelCol="Diabetes_binary",featuresCol="scaledFeatures",
                                      maxIter=100, layers=layers2, blockSize=64, seed=1234)

layers3 = [22, 8, 16, 32, 16, 8, 2]
mlp3 = MultilayerPerceptronClassifier(labelCol="Diabetes_binary",featuresCol="scaledFeatures",
                                      maxIter=100, layers=layers3, blockSize=64, seed=1234)


models = [lr1, lr2, lr3, rf1, rf2, rf3, mlp1, mlp2, mlp3]
best_model = None
best_pipeline = None
highest_F1 = 0.0

for model in models:
    # Creating Pipeline of transformation and prediction stages
    pipeline = Pipeline(stages=[vec_assembler, stdScaler,model])
    pipelineModel = pipeline.fit(train_df)

    predictions = pipelineModel.transform(test_df)

    evaluator = MulticlassClassificationEvaluator(labelCol='Diabetes_binary', predictionCol='prediction', metricName='f1')

    eval = evaluator.evaluate(predictions)
    print(f"F1 of {model} is = {eval}")

    if eval > highest_F1:
        highest_F1 = eval
        best_model = pipelineModel
        best_pipeline = pipeline


#save best model
print(best_model)
best_model.save("C:/Users/Administrator/PycharmProjects/domasna3/best_model/best_model")
best_pipeline.save("best_model/best_pipeline")