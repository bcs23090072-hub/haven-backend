import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Load Data (修复点：加上 encoding='utf-8' 以支持表情包)
print("Loading data from intents.json...")
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

patterns = []
tags = []

# Organize data into lists
for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)      # User sentences
        tags.append(intent['tag'])    # Class/Category

# 2. Vectorization (Convert text to numbers)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)

# 3. Training
print("Training the model...")
clf = LogisticRegression()
clf.fit(X, tags)

# 4. Save Model
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("✅ Training Complete!")
print("Generated 'model.pkl' and 'vectorizer.pkl' successfully.")