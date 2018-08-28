# -*- coding: utf-8 -*-

__author__ = "Ariel Rodrigues"
__version__ = "0.1.0"
__license__ = ""

"""
Module Docstring

params: {
    join_data: string,
    embeddings: string
}
"""

import luigi
import logging
import datetime
import utils

log = logging.getLogger(__name__)


class CNN(luigi.Task):
    params = luigi.DictParameter(default=None)

    def are_valid_params(self):
        return self.params and \
               type(self.params["join_data"]) is str and \
               type(self.params["embeddings"]) is str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.are_valid_params():
            raise Exception(f'CNN: worng params type')
        self.constants = {
            'join_data': f'../outputs/{self.params["join_data"]}',
            'embeddings': f'../outputs/{self.params["embeddings"]}'
        }

    def run(self):
        self.emite_log(f'starting task with params {str(self.constants)}')
        RESULTS = utils.main(self.constants['join_data'], self.constants['embeddings'])
        self.write_result(RESULTS)
        self.emite_log(f'task has finnished')


    def emite_log(self, message):
        formated_datetime = datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        log.info(f'{formated_datetime}: {message}')


    def output(self):
        return luigi.LocalTarget(f'../outputs/results/result_{self.params["join_data"]}_{self.params["embeddings"]}')


    def write_result(self, result):
        with self.output().open('w') as out_file:
            for line in result:
                out_file.write(line)


if __name__ == '__main__':
    luigi.run()

