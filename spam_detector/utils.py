def vectorize(model, tokenized_message):
    vectorized_message = []
    for text in tokenized_message:
        vectors = [model.wv[word] for word in text if word in model.wv]
        if vectors:
            vectorized_message.append(sum(vectors) / len(vectors))
        else:
            vectorized_message.append([])
    return vectorized_message
