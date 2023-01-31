import json
import time

import pandas
from kafka import KafkaProducer
import random


producer = KafkaProducer(bootstrap_servers='localhost:9092', security_protocol="PLAINTEXT")
data = pandas.read_csv('online.csv', index_col=0).drop('Diabetes_binary', axis=1)
#print(data.head())

for index, row in data.iterrows():

    producer.send(
        topic="health_data",
        value=row.to_json().encode("utf-8")
    )
    print(row)
    time.sleep(random.randint(500, 2000) / 1000.0)