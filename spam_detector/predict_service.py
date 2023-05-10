from pickle import load

import click
from nltk import word_tokenize

from .utils import vectorize

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
        predicted_probably = classifier.predict_proba(vectorized_message)
        click.echo(f"The message is spam with probably {predicted_probably[1]}")
    else:
        predicted = classifier.predict(vectorized_message)
        click.echo(f"Message is {'' if predicted[0][0] == 1 else 'not'}spam")
