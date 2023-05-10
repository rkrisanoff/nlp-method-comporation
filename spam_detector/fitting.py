import os

import nltk
import numpy as np
import pandas as pd

import string
from pickle import dump
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords

from gensim.models import Word2Vec
from gensim.models import FastText
from nltk.tokenize import word_tokenize

from .utils import vectorize

import shutil


def process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean


def fit_MNB(dataset):
    model_bag_of_words = CountVectorizer(analyzer=process)
    message_bag_of_words = model_bag_of_words.fit_transform(dataset['text'])
    # %%

    return model_bag_of_words, message_bag_of_words


def fit_word2text(tokenized_text):
    model = Word2Vec(tokenized_text, min_count=1)
    return model, vectorize(model, tokenized_text)


def fit_fast_text(tokenized_text):
    model = FastText(tokenized_text, min_count=1)
    return model, vectorize(model, tokenized_text)


def create_model_dir_struct():
    if os.path.exists("models") and os.path.isdir("models"):
        shutil.rmtree("models")
    os.makedirs("models")
    os.chdir("models")
    if not os.path.exists("vectorizers"):
        os.makedirs("vectorizers")
    if not os.path.exists("classifiers"):
        os.makedirs("classifiers")
        os.chdir("classifiers")
        os.makedirs("bag_of_words")
        os.makedirs("fast_text")
        os.makedirs("word2vec")
    os.chdir("../..")


def prepare_and_fit(dataset_file_csv='dataset/emails.csv'):
    nltk.download('stopwords')
    nltk.download('punkt')

    create_model_dir_struct()
    dataset = pd.read_csv(dataset_file_csv)
    # Check for duplicates and remove them
    dataset.drop_duplicates(inplace=True)
    # Fit the CountVectorizer to data
    tokenized_text = [word_tokenize(text) for text in dataset['text']]

    model_bag_of_words, message_bag_of_words = fit_MNB(dataset)
    model_word2text, message_word2vec = fit_word2text(tokenized_text)
    model_fast_text, message_fast_text = fit_fast_text(tokenized_text)

    minimal = min([min(vec) for vec in message_fast_text])
    if minimal < 0:
        minimal = np.abs(minimal)
        message_fast_text_non_negative = [vec + minimal for vec in message_fast_text]
    else:
        message_fast_text_non_negative = message_fast_text

    minimal = min([min(vec) for vec in message_word2vec])
    if minimal < 0:
        minimal = np.abs(minimal)
        message_word2vec_non_negative = [vec + minimal + np.abs(0.1) for vec in message_word2vec]
    else:
        message_word2vec_non_negative = message_word2vec
    dump(model_bag_of_words, open("models/vectorizers/bag_of_words_vectorizer.pkl", "wb"))
    dump(model_fast_text, open("models/vectorizers/fast_text_vectorizer.pkl", "wb"))
    dump(model_word2text, open("models/vectorizers/word2vec_vectorizer.pkl", "wb"))

    for vectorized_message, vectorizer_name in [
        (message_bag_of_words, "bag_of_words"),
        (message_fast_text_non_negative, "fast_text"),
        (message_word2vec_non_negative, "word2vec"),
    ]:
        x_train, x_test, y_train, y_test = train_test_split(vectorized_message, dataset['spam'], test_size=0.20,
                                                            random_state=0)
        print(vectorizer_name)
        for Model, model_name in [
            (MultinomialNB, "MultinomialNB"),
            (RandomForestClassifier, "RandomForestClassifier"),
            (SVC, "SVC"),
        ]:
            try:
                model = Model()
                model.fit(x_train, y_train)
                # Model predictions on test set
                y_pred = model.predict(x_test)
                # Model Evaluation | Accuracy
                accuracy = accuracy_score(y_test, y_pred)
                print(f"\taccuracy of {model_name}-> {accuracy * 100}")
                # Model Evaluation | Classification report
                dump(model, open(f"models/classifiers/{vectorizer_name}/{model_name}.pkl", 'wb'))

                print(classification_report(y_test, y_pred))

            except Exception as e:
                print(f"\t{model_name} ->"
                      f"\n\t {e}")
