
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import hstack
import warnings
warnings.filterwarnings("ignore")

# Load preprocessed dataset
df = pd.read_csv("dataset/WELFake_preprocessed.csv")

# Fill missing text if any
df['clean_text'].fillna('', inplace=True)

# Split into text and numerical features
X_text = df['clean_text']
X_numerical = df[['char_count', 'word_count', 'sentiment', 'readability']]
y = df['label']

# Encode label
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=10000)
X_text_vec = vectorizer.fit_transform(X_text)

# Scale numerical features to [0, 1] for chi2
scaler = MinMaxScaler()
X_numerical_scaled = scaler.fit_transform(X_numerical)

# Combine text and numerical features
X_combined = hstack([X_text_vec, X_numerical_scaled])

# Select best features using chi2
selector = SelectKBest(score_func=chi2, k=5000)
X_selected = selector.fit_transform(X_combined, y_encoded)

# Save selected features and labels
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_encoded, test_size=0.2, random_state=42)

# Save for model training later
import pickle
with open("X_train_selected.pkl", "wb") as f:
    pickle.dump(X_train, f)

with open("X_test_selected.pkl", "wb") as f:
    pickle.dump(X_test, f)

with open("y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)

with open("y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("selector.pkl", "wb") as f:
    pickle.dump(selector, f)

print("✅ Feature selection complete. Files saved for training.")

