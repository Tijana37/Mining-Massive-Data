import os, sys
#os.environ['PYSPARK_SUBMIT_ARGS'] = 'spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.3 StructuredStreaming.py'

from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import SparkSession

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

#Creating Spark app
spark = SparkSession \
    .builder \
    .appName("StructuredStreaming") \
    .getOrCreate()

#Loading the pipeline to transform the data
best_model = PipelineModel.load('best_model/best_pipeline')

#Function to be executed on every batch
def func_call(df, batch_id):
    # Cast from json
    df.selectExpr("CAST(value AS STRING) as json")
    values = df.rdd.map(lambda x: x.value).collect()
    predictions = best_model.transform(values)
    #Sending to output stream
    predictions.writeStream\
        .format("kafka") \
        .option("kafka.bootstrap.servers","localhost:9092")\
        .option("topic", "health_data_predicted")\
        .start()

#loading the stream of data from Kafka producer
df = spark \
  .readStream \
  .format("kafka") \
  .option("kafka.bootstrap.servers", "localhost:9092") \
  .option("subscribe", "health_data") \
  .load().trigger(processingTime="5 seconds").foreachBatch(func_call).start().awaitTermination()
