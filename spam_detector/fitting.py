import os

import warnings

import nltk
import numpy as np
import pandas as pd
import string
import time
from pickle import dump

from prettytable import PrettyTable
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import  accuracy_score, confusion_matrix
from nltk.corpus import stopwords

from gensim.models import Word2Vec
from gensim.models import FastText
from nltk.tokenize import word_tokenize

from .utils import vectorize

import shutil

warnings.simplefilter(action='ignore', category=FutureWarning)


def process(text):
    no_punc = [char for char in text if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    clean = [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]
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
    assert os.path.exists("models") and os.path.isdir("models"),"models directory doesn't exist!"
    for filename in os.listdir("models"):
        file_path = os.path.join("models", filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    os.chdir("models")
    os.makedirs("ru")
    os.makedirs("en")
    for lang in ["ru", "en"]:
        os.chdir(lang)
        os.makedirs("vectorizers")
        os.makedirs("classifiers")
        os.chdir("classifiers")
        os.makedirs("bag_of_words")
        os.makedirs("fast_text")
        os.makedirs("word2vec")
        os.chdir("../..")
    os.chdir("..")


def prepare_and_fit(datasets=None):
    if datasets is None:
        datasets = {"en": 'dataset/emails_en.csv', "ru": 'dataset/emails_ru.csv'}
    nltk.download('stopwords')
    nltk.download('punkt')
    create_model_dir_struct()
    for lang in ["ru", "en"]:  # "ru"
        print(f"fitting the model by lang={lang}")
        dataset = pd.read_csv(datasets[lang])
        # Check for duplicates and remove them
        dataset.drop_duplicates(inplace=True)
        # Fit the CountVectorizer to data
        tokenized_text = [word_tokenize(text) for text in dataset['text']]

        start_time = time.time()
        model_bag_of_words, message_bag_of_words = fit_MNB(dataset)
        finish_time = time.time()
        print(f"\tfitting bag of words, duration: {finish_time - start_time} seconds")

        start_time = time.time()
        model_word2text, message_word2vec = fit_word2text(tokenized_text)
        finish_time = time.time()
        print(f"\tfitting word to vec, duration: {finish_time - start_time} seconds")

        start_time = time.time()
        model_fast_text, message_fast_text = fit_fast_text(tokenized_text)
        finish_time = time.time()
        print(f"\tfitting fast text, duration: {finish_time - start_time} seconds")

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
        dump(model_bag_of_words, open(f"models/{lang}/vectorizers/bag_of_words_vectorizer.pkl", "wb"))
        dump(model_fast_text, open(f"models/{lang}/vectorizers/fast_text_vectorizer.pkl", "wb"))
        dump(model_word2text, open(f"models/{lang}/vectorizers/word2vec_vectorizer.pkl", "wb"))
        finish_time = time.time()
        print(finish_time - start_time)
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
                print(f"\n\n\n<= {lang}:{vectorizer_name}:{model_name} =>\n")
                try:

                    model = Model()

                    start_time = time.time()
                    model.fit(x_train, y_train)
                    finish_time = time.time()
                    print(f"fitting {model_name} by {vectorizer_name}: {finish_time - start_time} seconds")
                    # Model predictions on test set
                    y_pred = model.predict(x_test)
                    cm = confusion_matrix(y_test, y_pred)
                    table = PrettyTable()
                    print(f"TP = {cm[0][0]}\nFP = {cm[1][0]}\nTN = {cm[0][1]}\nFN = {cm[1][1]}\n")

                    table.field_names = ["Predicted \\ Actual", "Positive", "Negative"]
                    # добавление данных по одной строке за раз
                    table.add_row(["Positive", cm[0][0], cm[1][0]])
                    table.add_row(["Negative", cm[0][1], cm[1][1]])

                    print(f"{table}")

                    # Model Evaluation | Accuracy
                    accuracy = accuracy_score(y_test, y_pred)

                    print(f"accuracy of {model_name}-> {accuracy * 100}%")
                    # Model Evaluation | Classification report

                    dump(model, open(f"models/{lang}/classifiers/{vectorizer_name}/{model_name}.pkl", 'wb'))

                    # print(classification_report(y_test, y_pred))

                except Exception as e:
                    print(f"\t{model_name} ->"
                          f"\n\t {e}")
