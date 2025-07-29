from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to my Yelp Sentiment App!"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://10.0.0.138:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("best_sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

class Review(BaseModel):
    text: str

@app.post("/predict")
def predictSentiment(review: Review):
    X = vectorizer.transform([review.text])
    prediction = model.predict(X)[0]
    label = label_encoder.inverse_transform([prediction])[0]
    
    # Convert to native Python type before returning
    return {"prediction": str(label)}