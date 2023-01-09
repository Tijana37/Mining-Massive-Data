from dataclasses import dataclass, field
import pyspark
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.sql.types import IntegerType

import os
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Domasna2").getOrCreate()

sc = spark.sparkContext

df = spark.read.text("ml-100k/u.data")

df = df.selectExpr("split(value, '\t') as data")

df = df.withColumn('user', df['data'][0].cast(IntegerType()))
df = df.withColumn('item', df['data'][1].cast(IntegerType()))
df = df.withColumn('rating', df['data'][2].cast(IntegerType()))

df.show()

ratings = df.rdd.map(lambda l: Rating(user=l['user'],product=l['item'],rating=l['rating']))

ranks = [11, 12, 13]
iterationsnum = [11, 12, 13]
lambdas = [0.001, 0.01]   

mse_best = 100.0
rank_best = ranks[0]
iterations_best = iterationsnum[0]
lambda_best = lambdas[0]
# iterate for the given parameters, train and save a model based on them, save the parameters with the mse
for rank in ranks:
    for iterations in iterationsnum:
        for l in lambdas:
            model = ALS.train(ratings, rank, iterations, l)
            testdata = ratings.map(lambda p: (p[0], p[1]))
            predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))        
            ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
            mse = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
            print('{0} {1} {2}'.format(rank, iterations, l))
            print("Mean Squared Error = " + str(mse))
            # save only the best model by far
            if (mse < mse_best):
                mse_best = mse
                rank_best = rank
                iterations_best = iterations
                lambda_best = l
                model.save(sc, "models/model-"+str(rank_best)+"-"+str(iterations_best)+
                                    "-"+str(lambda_best)+"-"+str(mse_best))


print(mse_best)
best_model = MatrixFactorizationModel.load(sc, "models/model-"+str(rank_best)+"-"+str(iterations_best)+
            "-"+str(lambda_best)+"-"+str(mse_best))

# predictions from the loaded model
predictions = best_model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error For The Best Model = " + str(MSE))

