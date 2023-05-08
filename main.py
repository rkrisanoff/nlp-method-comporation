import click
from fitting import prepare_and_fit
from predict_service import predict


@click.command()
# @click.argument('dataset_file_csv')
def prepare_and_fit_wrapper():
    prepare_and_fit()


@click.command()
@click.option('--class-method', '-c')
@click.option('--vector-method', '-v')
@click.option('--message', '-m')
def predict_if_spam(vector_method, class_method, message):
    predict(
        classification_model_method=class_method,
        vectorizer_method=vector_method,
        message=message,
    )