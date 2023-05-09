import sys

import click
from nltk import word_tokenize

from .utils import vectorize
from pickle import load

vectorizers = {"bag_of_words": "bag_of_words", "fast_text": "fast_text", "word2vec": "word2vec"}
classifiers = {"naive_bayes": "MultinomialNB", "random_forest": "RandomForestClassifier", "svc": "SVC"}


@click.command()
@click.option('--vector-method', '-v', type=click.Choice(list(vectorizers.keys()), case_sensitive=False),
              required=True)
@click.option('--class-method', '-c', type=click.Choice(list(classifiers.keys()), case_sensitive=False),
              required=True)
@click.option('--probabilistic', "-p", "is_probabilistic", is_flag=True, default=False, show_default=True)
@click.argument('message', type=str)
def predict_if_spam(vector_method, class_method, message, is_probabilistic):
    vectorizer = load(
        open(f"models/vectorizers/{vectorizers[vector_method]}_vectorizer.pkl", "rb")
    )
    classifier = load(
        open(f"models/classifiers/{vectorizers[vector_method]}/{classifiers[class_method]}.pkl", 'rb'))

    if vector_method == "bag_of_words":
        vectorized_message = vectorizer.transform([message])
    else:
        tokenized_message = [word_tokenize(text) for text in [message]]
        vectorized_message = vectorize(vectorizer, tokenized_message)

    if is_probabilistic:
        click.echo(classifier.predict_proba(vectorized_message))
    else:
        click.echo(classifier.predict(vectorized_message))
