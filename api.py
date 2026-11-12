from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Lightweight Passenger Satisfaction API")

# Load the streamlined pipeline
model = joblib.load('models/model_pipeline.pkl')

class PassengerData(BaseModel):
    Type_of_Travel: str 
    Class: str
    Customer_Type: str 
    Inflight_wifi_service: int
    Online_boarding: int
    Inflight_entertainment: int
    Seat_comfort: int
    Leg_room_service: int

@app.get("/")
def home():
    return {"message": "API  is running."}

@app.post("/predict")
def predict_satisfaction(data: PassengerData):
    data_dict = data.model_dump()
    
    # Reconstruct CSV column names (replace underscores with spaces)
    data_dict = {k.replace('_', ' '): v for k, v in data_dict.items()}
        
    df = pd.DataFrame([data_dict])
    prediction = model.predict(df)
    
    return {"prediction": str(prediction[0])}