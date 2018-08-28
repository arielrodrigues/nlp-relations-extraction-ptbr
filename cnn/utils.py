import gzip
import pickle as pkl
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers import Dropout, GaussianNoise
from keras.layers import Embedding, concatenate
from keras.layers import Convolution1D, GlobalMaxPooling1D #, MaxPooling1D
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report


import scikitplot as skplt
import matplotlib.pyplot as plt

# Model Hyperparameters
HYPERPARAMETERS = {
    'filter_sizes': 3,
    'num_filters': 100,
    'max_num_words_in_vocabulary': 20000,
    'position_dims': 50
}

# Training parameters
TRAINING_PARAMETERS = {
    'batch_size': 64,
    'num_epochs': 1,
    'dropout_rate': 0,
    'std_noise': 0
}

RESULTS = []


def extract_gzip_data(infile):
    with gzip.open(infile, 'rb') as file:
        data = pkl.load(file)
        file.close()
    return data


def extract_matrices(infile):
    data = extract_gzip_data(infile)
    return data['train'], data['test']


def prepare_model(max_sequence_length, embedding_matrix, max_position, n_out):
    words_input = Input(shape=(max_sequence_length,), dtype='int32', name='words_input')
    words = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                      weights=[embedding_matrix], trainable=False)(words_input)

    distance1_input = Input(shape=(max_sequence_length,), dtype='int32', name='distance1_input')
    distance1 = Embedding(max_position, HYPERPARAMETERS['position_dims'])(distance1_input)

    distance2_input = Input(shape=(max_sequence_length,), dtype='int32', name='distance2_input')
    distance2 = Embedding(max_position, HYPERPARAMETERS['position_dims'])(distance2_input)

    output = concatenate([words, distance1, distance2])
    output = GaussianNoise(TRAINING_PARAMETERS['std_noise'])(output)
    output = Convolution1D(filters=HYPERPARAMETERS['num_filters'],
                           kernel_size=HYPERPARAMETERS['filter_sizes'],
                           padding='valid',
                           activation='relu',
                           strides=1)(output)
    output = GlobalMaxPooling1D()(output)
    output = Dropout(TRAINING_PARAMETERS['dropout_rate'])(output)
    output = Dense(n_out, activation='softmax')(output)

    model = Model(inputs=[words_input, distance1_input, distance2_input], outputs=[output])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, train_data):
    j = 1; k = 10; cvscores = []
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=1337)
    sentence_train, yLabel_train, positionMatrix1_train, positionMatrix2_train = train_data
    Y = np.argmax(yLabel_train, axis=-1)
    for train, test in skf.split(sentence_train, Y):
        # Fit the model
        RESULTS.append(f'train {k}: {j} / {k}')
        model.fit([sentence_train[train],
                   positionMatrix1_train[train],
                   positionMatrix2_train[train]],
                  yLabel_train[train],
                  epochs=TRAINING_PARAMETERS['num_epochs'],
                  batch_size=TRAINING_PARAMETERS['batch_size'],
                  verbose=1)
        # evaluate the model
        scores = model.evaluate([sentence_train[test], positionMatrix1_train[test], positionMatrix2_train[test]],
                                yLabel_train[test], verbose=0)
        RESULTS.append(f'val_acc: {scores[1] * 100}%')  # model.metrics_names[1]
        cvscores.append(scores[1] * 100)
        j = j + 1
    RESULTS.append(f'{np.mean(cvscores)}% (+/-{np.std(cvscores)})')


def main(join_data_file, embeddings_file):
    train_data, test_data = extract_matrices(join_data_file)
    sentence_train, yLabel_train, positionMatrix1_train, positionMatrix2_train = train_data
    sentence_test, yLabel_test, positionMatrix1_test, positionMatrix2_test = test_data

    max_position = max(np.max(positionMatrix1_train), np.max(positionMatrix2_train)) + 1
    n_out = yLabel_train.shape[1]
    max_sequence_length = sentence_train.shape[1]

    embedding_matrix = np.load(open(embeddings_file, 'rb'))

    # prepare and test model
    model = prepare_model(max_sequence_length, embedding_matrix, max_position, n_out)
    train_model(model, train_data)

    # test model
    predicted_yLabel = model.predict(
        [sentence_test, positionMatrix1_test, positionMatrix2_test],
        batch_size=None,
        verbose=0,
        steps=None
    )

    predicted_yLabel = np.argmax(predicted_yLabel, axis=-1)
    yLabel_test = np.argmax(yLabel_test, axis=-1)

    print(f'{yLabel_test}, {predicted_yLabel}')
    RESULTS.append(f'Classification report: \n {classification_report(yLabel_test, predicted_yLabel)}')

    return '\n'.join(RESULTS)
