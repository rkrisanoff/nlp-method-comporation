import os
from pickle import load

import click
from nltk import word_tokenize

from .utils import vectorize
from .domen import vectorizers, classifiers, langs

@click.command()
@click.option('--vector-method', '-v', type=click.Choice(list(vectorizers.keys()), case_sensitive=False),
              required=True)
@click.option('--class-method', '-c', type=click.Choice(list(classifiers.keys()), case_sensitive=False),
              required=True)
@click.option('--lang', '-l', type=click.Choice(list(langs.keys()), case_sensitive=False),
              required=True)
@click.option('--probabilistic', "-p", "is_probabilistic", is_flag=True, default=False, show_default=True)
@click.argument('message', type=str)
def predict_if_spam(vector_method, class_method, lang, message, is_probabilistic):
    if class_method == "svc" and is_probabilistic:
        click.echo("SVC doesn't support probabilistic prediction")

    vectorizer_path = [
        'models',
        lang,
        'vectorizers',
        f"{vectorizers[vector_method]}_vectorizer.pkl"
    ]
    with open(os.path.join(*vectorizer_path), "rb") as vectorizer_file:
        vectorizer = load(vectorizer_file)

    classifier_path = [
        'models',
        lang,
        'classifiers',
        vectorizers[vector_method],
        f"{classifiers[class_method]}.pkl",
    ]
    with open(os.path.join(*classifier_path), "rb") as classifier_file:
        classifier = load(classifier_file)

    if vector_method == "bag_of_words":
        vectorized_message = vectorizer.transform([message])
    else:
        tokenized_message = [word_tokenize(text) for text in [message]]
        vectorized_message = vectorize(vectorizer, tokenized_message)

    if is_probabilistic:
        predicted_probably = classifier.predict_proba(vectorized_message)
        click.echo(f"The message is spam with probably {predicted_probably[0][1]}")
    else:
        predicted = classifier.predict(vectorized_message)
        click.echo(f"Message is {'' if predicted[0] == 1 else 'not'} spam")
