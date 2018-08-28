import json
import gzip
import numpy as np
import pickle as pkl
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

np.random.seed(1337)


def parse_infile_line(line):
    data = line.strip().split('\t')
    sentence = data[3]
    label = data[0]
    return sentence, label


def extract_sentences_and_labels_from_files(files):
    sentences = []
    labels = []
    for infile in files:
        with open(infile) as file:
            for line in file:
                sentence, label = parse_infile_line(line)
                labels.append(label)
                sentences.append(sentence)
            file.close()
    return sentences, labels


def remove_duplicates_from_list(inlist):
    return list(set(inlist))


def enumerate_list(inlist):
    return dict((item, index) for index, item in enumerate(inlist))


def one_hot_from_dict(dictionary):
    result = dict()
    for item, index in dictionary.items():
        one_hot = [0] * len(dictionary)
        one_hot[index] = 1
        result[item] = one_hot
    return result


def extract_word_and_coefs_from_embedding(embedding):
    data = embedding.split()
    word = data[0]
    coefs = np.asarray(data[1:], dtype='float32')
    return word, coefs


def transform_embeddings_in_dictionary(embeddings_file):
    index = dict()
    coefs = None
    with open(embeddings_file) as embeddings:
        for embedding in embeddings:
            word, coefs = extract_word_and_coefs_from_embedding(embedding)
            index[word] = coefs
        embeddings.close()
    return index, coefs


def get_embeddings_matrix_dimensios(num_max_tokens, more_common_tokens_enum, coefs):
    embeddings_dimesion = len(coefs)
    number_of_tokens = min(num_max_tokens, len(more_common_tokens_enum))
    scale = 1 / np.sqrt(embeddings_dimesion)
    return number_of_tokens, embeddings_dimesion, scale


def generate_embeddings_matrix(pretrained_word_embeddings, num_max_tokens, more_common_tokens_enum):
    embeddings_dict, coefs = transform_embeddings_in_dictionary(pretrained_word_embeddings)
    print(f'Foram carregados {len(embeddings_dict)} vetores de palavras')
    lines, columns, scale = get_embeddings_matrix_dimensios(num_max_tokens, more_common_tokens_enum, coefs)
    embeddings_matrix = np.random.normal(0, scale=scale, size=[lines + 1, columns])
    for word, index in more_common_tokens_enum.items():
        if index >= lines:
            continue
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            embeddings_matrix[index] = embedding_vector[0:len(coefs)]
    print("Created vocab. with " + str(embeddings_matrix.shape[0]), " words.")
    return embeddings_matrix


def generate_distance_mapping(min_distance, max_distance):
    distance_mapping = {'PADDING': 0, 'LowerMin': 1, 'GreaterMax': 2}
    for dis in range(min_distance, max_distance + 1):
        distance_mapping[dis] = len(distance_mapping)
    return distance_mapping


def extract_tokens_from_sentence(sentence):
    tokens = sentence.strip().split(' ')
    while '' in tokens:
        tokens.remove('')
    return tokens


def extract_data_from_mapLabels(mapLabels, line):
    splits = line.split('\t')
    label = mapLabels[splits[0]]
    pos1 = splits[1]
    pos2 = splits[2]
    sentence = splits[3].lower()
    tokens = extract_tokens_from_sentence(sentence)
    return label, pos1, pos2, tokens


def find_distance(distance, min_distance, distance_mapping):
    if distance in distance_mapping:
        return distance_mapping[distance]
    elif distance <= min_distance:
        return distance_mapping['LowerMin']
    else:
        return distance_mapping['GreaterMax']


def generate_input_matrix(dataset_file, word2Idx, mapLabels, maxSentenceLen=100):
    print('Creating matrices for the sentence of %s ...' % dataset_file)
    min_distance = -40
    max_distance = 40
    distance_mapping = generate_distance_mapping(min_distance, max_distance)

    labels = []
    position_matrix_1 = []
    position_matrix_2 = []
    token_matrix = []

    for line in open(dataset_file):
        label, pos1, pos2, tokens = extract_data_from_mapLabels(mapLabels, line)

        token_ids = np.zeros(maxSentenceLen)
        position_values1 = np.zeros(maxSentenceLen)
        position_values2 = np.zeros(maxSentenceLen)

        for idx in range(0, min(maxSentenceLen, len(tokens))):
            token_ids[idx] = word2Idx.word_index[tokens[idx]]
            position_values1[idx] = find_distance(idx - int(pos1), min_distance, distance_mapping)
            position_values2[idx] = find_distance(idx - int(pos2), min_distance, distance_mapping)

        token_matrix.append(token_ids)
        position_matrix_1.append(position_values1)
        position_matrix_2.append(position_values2)
        labels.append(label)

    return np.array(labels, dtype='int32'), \
           np.array(token_matrix, dtype='int32'), \
           np.array(position_matrix_1, dtype='int32'), \
           np.array(position_matrix_2, dtype='int32')


def split_in_train_test_matrices(input_matrix):
    y, sentence, pos1, pos2 = input_matrix

    sentence_train, sentence_test, \
    yLabel_train, yLabel_test, \
    positionMatrix1_train, positionMatrix1_test, \
    positionMatrix2_train, positionMatrix2_test = train_test_split(sentence, y, pos1, pos2, train_size=0.7, stratify=y)
    del sentence, y, pos1, pos2

    return [sentence_train, yLabel_train, positionMatrix1_train, positionMatrix2_train], \
           [sentence_test, yLabel_test, positionMatrix1_test, positionMatrix2_test]


def generate_train_test_matrices(dataset_file, word2Idx, mapLabels, maxSentenceLen=100):
    input_matrix = generate_input_matrix(dataset_file, word2Idx, mapLabels, maxSentenceLen)
    train_matrix, test_matrix = split_in_train_test_matrices(input_matrix)

    del input_matrix
    return {'train': train_matrix, 'test': test_matrix}


def get_labels_in_onehot(labels):
    labels_without_duplicates = remove_duplicates_from_list(labels)
    enum_labels = enumerate_list(labels_without_duplicates)
    print(f'Labels: {enum_labels}')
    print(f'One-hot labels: {one_hot_from_dict(enum_labels)}')
    return one_hot_from_dict(enum_labels)
