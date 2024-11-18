import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Data Preprocessing
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return stemmed_tokens

# 2. Feature Engineering
def create_word_embeddings(sentences):
    model = Word2Vec(sentences, min_count=1, size=100, window=5)
    word_vectors = model.wv
    return word_vectors

def sentence_embedding(sentence, word_vectors):
    words = word_tokenize(sentence)
    words = [word for word in words if word in word_vectors.vocab]
    if not words:
        return np.zeros(100)
    word_vecs = [word_vectors[word] for word in words]
    sentence_vec = np.mean(word_vecs, axis=0)
    return sentence_vec

# 3. Model Selection and Training (In this case, we'll use cosine similarity for matching)
def match_sentences(sentence1, sentence2, word_vectors):
    vec1 = sentence_embedding(sentence1, word_vectors)
    vec2 = sentence_embedding(sentence2, word_vectors)
    similarity = cosine_similarity([vec1], [vec2])[0][0]
    return similarity

# Load the dataset
df = pd.read_csv("clean_mh.csv")

# Preprocess the text
df['processed_text'] = df['sentence'].apply(preprocess_text)

# Create word embeddings
word_vectors = create_word_embeddings(df['processed_text'])

# Match sentences based on similarity
for index, row in df.iterrows():
    sentence1 = row['processed_text']
    for other_index, other_row in df.iterrows():
        if index != other_index:
            sentence2 = other_row['processed_text']
            similarity = match_sentences(sentence1, sentence2, word_vectors)
            if similarity > 0.8:  # Adjust threshold as needed
                print(f"Sentence 1: {sentence1}")
                print(f"Sentence 2: {sentence2}")
                print(f"Similarity: {similarity}\n")