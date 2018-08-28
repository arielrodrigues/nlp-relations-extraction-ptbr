# -*- coding: utf-8 -*-

__author__ = "Ariel Rodrigues"
__version__ = "0.1.0"
__license__ = ""

"""
Module Docstring

embedding: {
  name: string,
  input_path: string
}
"""

import luigi
import logging
import datetime
import utils

log = logging.getLogger(__name__)


def emite_log(message):
    formated_datetime = datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
    log.info(f'{formated_datetime} - PreprocessWordEmbeddings: {message}')


class PreprocessWordEmbeddings(luigi.Task):
    embedding = luigi.DictParameter(default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.embedding:
            raise Exception(f'PreprocessWordEmbeddings: can\'t run without a embedding')

    def run(self):
        emite_log(f'has started for {self.embedding["name"]}')
        with open(self.embedding["input_path"], 'r') as infile:
            embeddings = utils.clean_embedding(infile)
            infile.close()
        self.write_result(embeddings)

    def write_result(self, result):
        with self.output().open('w') as out_file:
            for line in result:
                out_file.write(line)
        self.output()
        emite_log(f'has finnished for {self.embedding["name"]}')

    def output(self):
        formated_datetime = datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        return luigi.LocalTarget(f'../outputs/{self.embedding["name"]}_preprocessed_{formated_datetime}')


if __name__ == '__main__':
    luigi.run()

