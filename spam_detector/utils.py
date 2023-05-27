"""
The package with useful utils
"""
import numpy as np


def vectorize(model, tokenized_message):
    """
    method for transform the tokenized message to vector representation
    """
    vectorized_message = []
    for text in tokenized_message:
        vectors = [model.wv[word] for word in text if word in model.wv]
        if vectors:
            vectorized_message.append(sum(vectors) / len(vectors))
        else:
            vectorized_message.append([])

    minimal = min([min(vec) for vec in vectorized_message])
    if minimal < 0:
        minimal = np.abs(minimal)
        message_non_negative = [vec + minimal + np.abs(0.1) for vec in vectorized_message]
    else:
        message_non_negative = vectorized_message

    return message_non_negative
