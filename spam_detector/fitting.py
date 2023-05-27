import os
import sys

import warnings
from typing import NamedTuple

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

from sklearn.metrics import accuracy_score, confusion_matrix
from nltk.corpus import stopwords

from gensim.models import Word2Vec
from gensim.models import FastText
from nltk.tokenize import word_tokenize

from dataclasses import dataclass

from .utils import vectorize

import shutil

warnings.simplefilter(action='ignore', category=FutureWarning)


@dataclass
class VectorizerFitReport:
    id: str | None
    lang: str
    title: str
    duration: float
    size: int


@dataclass
class ClassifierModelMetrics:
    TN: int
    FP: int
    FN: int
    TP: int


@dataclass
class ClassifierFitReport:
    lang: str
    title: str
    vectorizer_id: str
    duration: float
    size: int
    metrics: ClassifierModelMetrics
    accuracy: float


def process(text):
    no_punc = [char for char in text if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    clean = [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]
    return clean


def fit_bag_of_words(dataset):
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
    if os.path.exists("models") and os.path.isdir("models"):  # , "models directory doesn't exist!"
        for filename in os.listdir("models"):
            file_path = os.path.join("models", filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    else:
        os.mkdir("models")
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


def form_md_report(reports: list[VectorizerFitReport | ClassifierFitReport]) -> str:
    def vectorizer_report_item(vfr: VectorizerFitReport) -> str:
        return f"- {vfr.title}, duration of fitting: {vfr.duration} seconds\n"

    def classifier_report_item(cfr: ClassifierFitReport) -> str:
        table = PrettyTable()
        table.field_names = ["Predicted \\ Actual", "Positive", "Negative"]
        table.add_row(["Positive", cfr.metrics.TP, cfr.metrics.FN])
        table.add_row(["Negative", cfr.metrics.FP, cfr.metrics.TN])

        return f"""
#### {cfr.title} by {cfr.vectorizer_id}

Duration of fitting: {cfr.duration} seconds

```                       
{table}
```
                       
Accuracy of {cfr.title}-> {cfr.accuracy}%
"""

    return f"""
# Report
    
## English
    
### Vectorizers
    
{''.join(map(vectorizer_report_item,
             filter(lambda report: isinstance(report, VectorizerFitReport) and report.lang == "en",
                    reports)))}
    
### Classifiers
    
{''.join(map(classifier_report_item,
             filter(lambda report: isinstance(report, ClassifierFitReport) and report.lang == "en",
                    reports)))}
    
## Russian
    
### Vectorizers
    
{''.join(map(vectorizer_report_item,
             filter(lambda report: isinstance(report, VectorizerFitReport) and report.lang == "ru",
                    reports)))}
    
### Classifiers
    
{''.join(map(classifier_report_item,
             filter(lambda report: isinstance(report, ClassifierFitReport) and report.lang == "ru",
                    reports)))}
    
"""


def prepare_and_fit(datasets=None):
    if datasets is None:
        datasets = {"en": 'dataset/emails_en.csv', "ru": 'dataset/emails_ru.csv'}
    create_model_dir_struct()
    reports = list()
    for lang in ["ru", "en"]:
        dataset = pd.read_csv(datasets[lang])
        # dataset = pd.concat([dataset[dataset["spam"] == 1].head(100), dataset[dataset["spam"] == 0].head(100)])
        # Check for duplicates and remove them
        dataset.drop_duplicates(inplace=True)
        # Fit the CountVectorizer to data
        tokenized_text = [word_tokenize(text) for text in dataset['text']]
        start_time = time.time()
        model_bag_of_words, message_bag_of_words = fit_bag_of_words(dataset)
        finish_time = time.time()
        reports.append(VectorizerFitReport(
            id="bag_of_words",
            lang=lang,
            title="Bag of words",
            duration=finish_time - start_time,
            size=sys.getsizeof(model_bag_of_words)
        ))

        start_time = time.time()
        model_word2text, message_word2vec = fit_word2text(tokenized_text)
        finish_time = time.time()
        reports.append(VectorizerFitReport(
            id="word2vec",
            lang=lang,
            title="Word to vector",
            duration=finish_time - start_time,
            size=sys.getsizeof(model_word2text),
        ))

        start_time = time.time()
        model_fast_text, message_fast_text = fit_fast_text(tokenized_text)
        finish_time = time.time()
        reports.append(VectorizerFitReport(
            id="fast_text",
            lang=lang,
            title="Fast Text",
            duration=finish_time - start_time,
            size=sys.getsizeof(model_fast_text),
        ))
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
        for vectorized_message, vectorizer_name in [
            (message_bag_of_words, "bag_of_words"),
            (message_fast_text_non_negative, "fast_text"),
            (message_word2vec_non_negative, "word2vec"),
        ]:
            x_train, x_test, y_train, y_test = train_test_split(vectorized_message, dataset['spam'], test_size=0.20,
                                                                random_state=0)
            for Model, model_id, model_name in [
                (MultinomialNB, "MultinomialNB", "Naive Bayes"),
                (RandomForestClassifier, "RandomForestClassifier", "Random forest classifier"),
                (SVC, "SVC", "Support Vector Classification"),
            ]:
                try:
                    model = Model()

                    start_time = time.time()
                    model.fit(x_train, y_train)
                    finish_time = time.time()
                    # Model predictions on test set
                    y_pred = model.predict(x_test)
                    cm = confusion_matrix(y_test, y_pred)
                    tn, fp, fn, tp = cm.ravel()

                    # Model Evaluation | Accuracy
                    accuracy = accuracy_score(y_test, y_pred)

                    # Model Evaluation | Classification report

                    reports.append(ClassifierFitReport(
                        lang=lang,
                        title=model_name,
                        vectorizer_id=vectorizer_name,
                        duration=finish_time - start_time,
                        size=sys.getsizeof(model),
                        metrics=ClassifierModelMetrics(
                            TP=tp, TN=tn, FP=fp, FN=fn,
                        ),
                        accuracy=accuracy * 100,
                    ))
                    dump(model, open(f"models/{lang}/classifiers/{vectorizer_name}/{model_id}.pkl", 'wb'))

                except Exception as e:
                    print(f"\t{model_name} of {vectorizer_name} ->"
                          f"\n\t {e}")

    with open("./report.md", "w") as report_file:
        report_file.write(form_md_report(reports))
