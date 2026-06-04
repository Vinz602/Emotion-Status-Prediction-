from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()
# Load the machine learning model
pipeline = joblib.load('emotion_status_prediction_pipeline.pkl')
le = joblib.load("label_encoder.pkl")

class emotionFeatures(BaseModel):
    Age : int
    Gender : str
    Daily_Usage_Time_minutes : int
    Posts_Per_Day : int
    Likes_Received_Per_Day : int
    Comments_Received_Per_Day : int
    Messages_Sent_Per_Day : int
    Platform : str
    
@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model API"}

@app.post('/predict')

def predict(emotion: emotionFeatures):
    
    #model expect data sudah diencode, lakukan encoding disini juga   
    data = {
        'Age': emotion.Age,
        'Gender': emotion.Gender,
        'Daily_Usage_Time_minutes': emotion.Daily_Usage_Time_minutes,
        'Posts_Per_Day': emotion.Posts_Per_Day,
        'Likes_Received_Per_Day': emotion.Likes_Received_Per_Day,
        'Comments_Received_Per_Day': emotion.Comments_Received_Per_Day,
        'Messages_Sent_Per_Day': emotion.Messages_Sent_Per_Day,
        'Platform': emotion.Platform
    }

    input_df = pd.DataFrame([data])
    
    try:
        prediction = pipeline.predict(input_df)
        # Decode the prediction back to the original label
        decoded_prediction = le.inverse_transform(prediction)
        return {"prediction": decoded_prediction[0]}

    except Exception as e:
        return {"error": str(e)}