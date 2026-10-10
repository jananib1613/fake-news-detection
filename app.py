from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np
import nltk
import textstat
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack

# Download necessary NLTK data
nltk.download('vader_lexicon')

# Flask setup
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Label encoder (for consistency; not used in prediction directly here)
label_encoder = LabelEncoder()
label_encoder.fit(["FAKE", "REAL"])

@app.route("/", methods=["GET", "POST"])
def home():
    if "history" not in session:
        session["history"] = []

    prediction = None
    confidence = None
    sentiment = None

    if request.method == "POST":
        title = request.form["title"]
        content = request.form["content"]
        full_text = title + " " + content

        # Numerical features
        char_count = len(full_text)
        word_count = len(full_text.split())
        sentiment_score = sentiment_analyzer.polarity_scores(full_text)["compound"]
        readability = textstat.flesch_reading_ease(full_text)

        # Assign sentiment label
        if sentiment_score >= 0.05:
            sentiment = "Positive"
        elif sentiment_score <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        # Transform text
        vec = vectorizer.transform([full_text])

        # Combine TF-IDF and numerical features
        numerical = np.array([[char_count, word_count, sentiment_score, readability]])
        final_input = hstack([vec, numerical])

        # Predict
        pred_proba = model.predict_proba(final_input)[0]
        pred_label_index = np.argmax(pred_proba)
        pred_label = model.classes_[pred_label_index]
        confidence = round(pred_proba[pred_label_index] * 100, 2)
        label_map = {0: "FAKE", 1: "REAL"}
        prediction = label_map[int(pred_label)]


        # Save to history
        session["history"].append({
            "title": title,
            "prediction": prediction,
            "confidence": confidence
        })
        session.modified = True

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           sentiment=sentiment,
                           history=session["history"])

@app.route("/clear", methods=["POST"])
def clear():
    session["history"] = []
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)
