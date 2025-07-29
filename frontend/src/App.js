import React, { useState } from 'react';
import './App.css';

function App(){
  const[review, setReview] = useState('');
  const[result, setResult] = useState(null);

  const handlePredict = async() => {
    const response = await fetch('http://127.0.0.1:8000/predict/',{
      method: 'POST',
      headers: {
        'Content-Type' : 'application/json',
      },
      body: JSON.stringify({ text: review }),

    });

    if(response.ok){
      const data = await response.json();
      setResult(data.prediction);
    }else{
      setResult('Error occurred.');
    }
  };

  return (
    <div className="app-container">
      <h1 className = "heading"> Yelp Review Sentiment Predictor</h1>
      <textarea
        rows="5"
        cols="60"
        placeholder="Hello, enter your Yelp review here..."
        value={review}
        onChange={(e) => setReview(e.target.value)}
      />
      <br />
      <button onClick={handlePredict}>Predict Sentiment</button>
      {result && <p>Prediction: <strong>{result}</strong></p>}
    </div>
  );
}


export default App;

