from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text(text_data):
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
    return tfidf_matrix, tfidf_vectorizer
