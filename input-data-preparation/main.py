# -*- coding: utf-8 -*-

__author__ = "Ariel Rodrigues"
__version__ = "0.1.0"
__license__ = ""

"""
Module Docstring

params: {
    dataset: string,
    embeddings: string,
    max_tokens: number,
    max_len: number,
}
"""

import luigi
import logging
import datetime
import gzip
import pickle as pkl
import numpy as np
import utils
from keras.preprocessing.text import Tokenizer

log = logging.getLogger(__name__)


class PrepareInputData(luigi.Task):
    params = luigi.DictParameter(default=None)

    def are_valid_params(self):
        return self.params and \
               type(self.params["dataset"]) is str and \
               type(self.params["embeddings"]) is str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.are_valid_params():
            raise Exception(f'PrepareInputData: worng params type')
        self.constants = {
            'dataset': f'../outputs/{self.params["dataset"]}',
            'embeddings': f'../outputs/{self.params["embeddings"]}',
            'num_max_tokens': self.params["max_tokens"] if type(self.params["max_tokens"]) is int else 500000,
            'sentence_max_lenght': self.params["max_len"] if type(self.params["max_len"]) is int else 50
        }

    def run(self):
        self.emite_log(f'starting task with params {str(self.constants)}')
        sentences, labels = utils.extract_sentences_and_labels_from_files([self.constants['dataset']])
        labels_onehot = utils.get_labels_in_onehot(labels)

        keras_tokenizer = Tokenizer(num_words=self.constants['num_max_tokens'], filters='\t\n')
        keras_tokenizer.fit_on_texts(sentences)

        embeddings_matrix = utils.generate_embeddings_matrix(
            self.constants['embeddings'],
            self.constants['num_max_tokens'],
            keras_tokenizer.word_index
        )
        np.save(open(self.output_embeddings_matrix().path, 'wb'), embeddings_matrix)

        join_matrix = utils.generate_train_test_matrices(
            self.constants['dataset'],
            keras_tokenizer,
            labels_onehot,
            self.constants['sentence_max_lenght']
        )
        pkl.dump(join_matrix, gzip.open(self.output_join_matrix().path, 'wb'))

        self.emite_log(f'task has finnished')

    def emite_log(self, message):
        formated_datetime = datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        log.info(f'{formated_datetime}: {message}')

    def output_embeddings_matrix(self):
        return luigi.LocalTarget(f'../outputs/{self.params["embeddings"]}_matrix.npz', format=luigi.format.Gzip)

    def output_join_matrix(self):
        return luigi.LocalTarget(f'../outputs/ready_{self.params["dataset"]}_{self.params["embeddings"]}_matrices.pkl.gz', format=luigi.format.Gzip)


if __name__ == '__main__':
    luigi.run()

