import pandas as pd
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_excel("dataset.xlsx")

# Map ratings to sentiment column if it doesn't exist
if 'sentiment' not in df.columns:
    def map_sentiment(rating):
        if rating <= 2:
            return "Negative"
        elif rating == 3:
            return "Neutral"
        else:
            return "Positive"
    df['sentiment'] = df['rating'].apply(map_sentiment)

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

# Create combined text column for preprocessing if doesn't exist
if 'text_preprocessed' not in df.columns:
    df['text'] = df['body'].fillna('') + ' ' + df['title'].fillna('')
    df['text_preprocessed'] = df['text'].apply(preprocess_text)

# Define features and target
X = df['text_preprocessed']
y = df['sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel='linear', probability=True),
    "RandomForest": RandomForestClassifier(),
    "DecisionTree": DecisionTreeClassifier()
}

# Train and save models
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_vec, y_train)
    with open(f"{name}_model.pkl", 'wb') as f:
        pickle.dump(model, f)

# Save vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("All models and vectorizer saved successfully.")
