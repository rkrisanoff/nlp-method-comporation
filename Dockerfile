FROM mwalbeck/python-poetry:1-3.11
LABEL authors="drukhary"

WORKDIR /app
COPY . /app

RUN \
    poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-cache --only main \
    && poetry run python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')" \
    && rm -rf /root/.cache/pypoetry

#VOLUME "${pwd}"/dataset:/app/datasets
#VOLUME "${pwd}"/models:/app/models


ENTRYPOINT ["poetry","run"]