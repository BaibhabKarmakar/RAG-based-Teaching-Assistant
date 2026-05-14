import requests
import os
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed" , json = {
        "model" : "bge-m3",
        "input" : text_list
    })
    embedding = r.json()['embeddings']
    return embedding

jsons = os.listdir("jsons")
my_dicts = []
chunk_id = 1
for file in jsons:
    with open(f"jsons/{file}") as f:
        content = json.load(f)

    print(f"Creating embeddings for {file}")
    embeddings = create_embedding([c['text'] for c in content['chunk']])
    for i,chunk in enumerate(content['chunk']):
        chunk['chunk_id'] = chunk_id
        chunk['embeddings'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)

df = pd.DataFrame.from_records(my_dicts)
# Saving the dataFrame to joblib : 
joblib.dump(df , 'embeddings.joblib')






