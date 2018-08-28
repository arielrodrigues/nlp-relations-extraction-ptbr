from functools import reduce


def str_to_float(str):
    try:
        result = float(str)
        return result
    except ValueError:
        return None


def list_itens_has_same_type(data, dtype):
    new_data = map(lambda item: str_to_float(item), data)
    return reduce(lambda prev, item: type(item) is dtype and prev, new_data, True)


def extract_word_and_coefs(embedding):
    data = embedding.split()
    word = data[0]
    coefs = data[1:]
    return word, coefs


def embedding_is_valid(embedding):
    word, coefs = extract_word_and_coefs(embedding)
    return type(word) is str and list_itens_has_same_type(coefs, float)


def remove_number_of_lines_and_coluns_from_embeddings(embeddings):
    return embeddings[1:]


def clean_embedding(dataset):
    embeddings = list(filter(embedding_is_valid, dataset))
    embeddings = remove_number_of_lines_and_coluns_from_embeddings(embeddings)
    return embeddings
