import pandas as pd

import string
from pickle import dump
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
import warnings

from gensim.models import Word2Vec
from gensim.models import FastText
from nltk.tokenize import word_tokenize

# warnings.simplefilter(action='ignore', category=FutureWarning)
# nltk.download('stopwords')
# nltk.download('punkt')


def process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean


def vectorize(model, tokenized_message):
    vectorized_message = []
    for text in tokenized_message:
        vectors = [model.wv[word] for word in text if word in model.wv]
        if vectors:
            vectorized_message.append(sum(vectors) / len(vectors))
        else:
            vectorized_message.append([])
    return vectorized_message


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


def prepare_and_fit(dataset_file_csv='dataset/emails.csv'):
    dataset = pd.read_csv(dataset_file_csv)
    # Check for duplicates and remove them
    dataset.drop_duplicates(inplace=True)
    # Fit the CountVectorizer to data
    tokenized_text = [word_tokenize(text) for text in dataset['text']]

    model_bag_of_words, message_bag_of_words = fit_MNB(dataset)
    model_word2text, message_word2vec = fit_word2text(tokenized_text)
    model_fast_text, message_fast_text = fit_fast_text(tokenized_text)

    dump(model_bag_of_words, open("vectorizators/model_bag_of_words.pkl", "wb"))
    dump(model_fast_text, open("vectorizators/model_fast_text.pkl", "wb"))
    dump(model_word2text, open("vectorizators/model_word2text.pkl", "wb"))

    for vectorizedMessage, vectorizator_name in [
        (message_bag_of_words, "bag_of_words"),
        (message_fast_text, "fast_text"),
        (message_word2vec, "word2vec"),
    ]:
        x_train, x_test, y_train, y_test = train_test_split(vectorizedMessage, dataset['spam'], test_size=0.20,
                                                            random_state=0)
        print(vectorizator_name)
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
                dump(model, open(f"models/{vectorizator_name}/{model_name}.pkl", 'wb'))

                print(classification_report(y_test, y_pred))

            except Exception as e:
                print(f"\t{model_name} ->"
                      f"\n\t {e}")
