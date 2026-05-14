import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed" , json = {
        "model" : "bge-m3",
        "input" : text_list
    })
    embedding = r.json()['embeddings']
    return embedding

def inference(prompt):
    response = client.chat.completions.create(
        model = "deepseek-reasoner",
        messages = [
            {
                "role" : "system",
                "content" : "You are a helpful assistant for a Machine Learning in Healthcare course."
            },
            {
                "role" : "user",
                "content" : prompt
            }
        ],
        stream = True
    )
    full_response = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
            full_response += chunk.choices[0].delta.content
        
    print()
    return full_response

df = joblib.load('embeddings.joblib')

incoming_query = input("Ask a Question : ")
# We can pass the list also that will be saving more time . 
question_embedding = create_embedding([incoming_query])[0]
# print("Question embeddings are : ",question_embedding)

# Find similarities of question_embedding with other embeddings : 

# making it a 2D vector type : 
# print(np.vstack(df['embeddings'].values))
# print(np.vstack(df['embeddings']).shape)

# finding simlarities with an returned array : 
similarities = cosine_similarity(np.vstack(df['embeddings']), [question_embedding]).flatten()
# print(similarities)

# Taking top relevant results: 
threshold = 0.4
top_results = 50
max_indx = similarities.argsort()[::-1][:top_results]
mex_indx = [i for i in max_indx if similarities[i] > threshold]
# print(max_indx)
new_df = df.loc[max_indx]

# Want to see the required things from the dataframe : 
# print(new_df[["title" , "number" , "text"]])

prompt = f''' I am teaching the machine learning concepts in the field of Healthcare . Here are video subtitle chunks containing video title , number , start time , end time , the text at that time : 

{new_df[["title" , "number" , "text" , "start" , "end"]].to_json(orient="records")}
Strictly don't follow this format for repsponse . User don't know about chunks and other things about the project . 
----------------------------------------------
"{incoming_query}"
User asked this question related to the video chunks . 
You have to answer in a human way where and how much content is taught where means in which video and at what timestamp (make it in hours,minutes and seconds so that timestamp is more readable for human being). 
Make sure that timestamp should be upto 1 decimal for seconds .
Guide the user to go to that particular video.
If user asked any unrelated question, tell him that you can only answer related to the course . 
'''

# for index,item in new_df.iterrows():
#     print(index, item["number"], item["text"] , item["title"] , item["start"] , item["end"])

answer = inference(prompt)
print("\n-------Answer--------\n")
print(answer) 

with open("response.txt" , "w" , encoding = "utf-8") as f:
    f.write(answer)