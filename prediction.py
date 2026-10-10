import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import textstat

# Load trained model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Prediction function
def predict_news(news_text):
    # Basic text stats
    char_count = len(news_text)
    word_count = len(news_text.split())
    sentiment = TextBlob(news_text).sentiment.polarity
    readability = textstat.flesch_reading_ease(news_text)

    # Vectorize the text
    text_vec = vectorizer.transform([news_text])

    # Combine with numerical features
    numerical = np.array([[char_count, word_count, sentiment, readability]])
    from scipy.sparse import hstack
    final_input = hstack([text_vec, numerical])

    # Predict
    pred = model.predict(final_input)[0]
    pred_proba = model.predict_proba(final_input)[0]

    label = "REAL" if pred == 1 else "FAKE"
    confidence = round(max(pred_proba) * 100, 2)

    return label, confidence

# Example usage
if __name__ == "__main__":
    sample_news = input("Enter news text to predict:\n")
    label, confidence = predict_news(sample_news)
    print(f"\n🧠 Prediction: {label}\n🎯 Confidence: {confidence}%")
