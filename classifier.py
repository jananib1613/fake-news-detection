import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import numpy as np

# Step 1: Load preprocessed dataset
df = pd.read_csv("dataset/WELFake_preprocessed.csv")

# Step 2: Drop rows with missing clean_text
df.dropna(subset=['clean_text'], inplace=True)

# Step 3: Separate features and labels
X_text = df['clean_text']
X_numerical = df[['char_count', 'word_count', 'sentiment', 'readability']]
y = df['label']

# Step 4: Train-test split
X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
    X_text, X_numerical, y, test_size=0.2, random_state=42
)

# Step 5: TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_text_train_vec = vectorizer.fit_transform(X_text_train)
X_text_test_vec = vectorizer.transform(X_text_test)

# Step 6: Combine text and numerical features
X_train_combined = hstack([X_text_train_vec, X_num_train])
X_test_combined = hstack([X_text_test_vec, X_num_test])

# Step 7: Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_combined, y_train)

# Step 8: Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model trained and saved successfully!")

