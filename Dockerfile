FROM python:3.11-slim
LABEL authors="drukhary"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSL https://install.python-poetry.org | python3.11 - \
    && echo 'export PATH="$HOME/.poetry/bin:$PATH"' >> ~/.bashrc

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app
COPY . /app

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-cache --only main \
    && poetry cache clear pypi --all -n \
    && rm -rf /root/.cache/pypoetry \
    && poetry run python -m pip install . \
    && python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

ENTRYPOINT ["poetry","run"]