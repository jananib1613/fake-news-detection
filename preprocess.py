import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('stopwords')
nltk.download('vader_lexicon')

stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = text.lower()
    words = text.split()
    return " ".join([word for word in words if word not in stop_words])

def extract_features(df):
    print("✅ Cleaning text...")
    df['clean_text'] = df['text'].apply(clean_text)

    print("✅ Calculating char and word count...")
    df['char_count'] = df['clean_text'].apply(len)
    df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))

    print("✅ Calculating sentiment...")
    df['sentiment'] = df['clean_text'].apply(lambda x: sia.polarity_scores(x)['compound'])

    return df

def preprocess_dataset(input_path, output_path):
    print("🔄 Loading dataset...")
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['text', 'label']).reset_index(drop=True)

    print("🔄 Sampling 25,000 rows...")
    df = df.sample(25000, random_state=42)

    print("⚙️ Extracting features...")
    df = extract_features(df)

    print("💾 Saving cleaned dataset...")
    df.to_csv(output_path, index=False)
    print(f"✅ Done! Saved to {output_path}")

if __name__ == "__main__":
    preprocess_dataset("dataset/WELFakeDataset.csv", "dataset/WELFake_preprocessed.csv")



