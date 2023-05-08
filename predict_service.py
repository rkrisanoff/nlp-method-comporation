from nltk import word_tokenize

from fitting import vectorize
from pickle import load
import string


def predict(vectorizer_method, classification_model_method, message):
    vectorizer = load(open(
        f"vectorizators/model_{vectorizer_method}.pkl", "rb"))
    classification_model = load(open(
        f"models/{vectorizer_method}/{classification_model_method}.pkl", 'rb'))
    tokenized_message = [word_tokenize(text) for text in [message]]
    vectorized_message = vectorize(vectorizer, tokenized_message)
    print(classification_model.predict_proba(vectorized_message))
