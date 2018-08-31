# -*- coding: utf-8 -*-

__author__ = "Ariel Rodrigues"
__version__ = "0.1.0"
__license__ = ""

"""
Module Docstring

dataset: {
  name: string
  path: string
  lines_per_sentence: number
}
"""

import luigi
import logging
import datetime
from preprocessing import utils

log = logging.getLogger(__name__)


class FormatDataset(luigi.Task):
    dataset = luigi.DictParameter(default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.toolset = utils.define_toolset(self.dataset["type"])
        if not self.dataset:
            raise Exception(f'FormatDataset: can\'t run without a dataset')
            return

    def run(self):
        self.emite_log(f'starting task for {self.dataset["name"]}')
        raw_lines = self.toolset.get_lines_from_file(self.dataset["path"])
        print(f'Dataset has {len(raw_lines)} lines')
        clean_dataset = self.toolset.clean_dataset(raw_lines, self.dataset["lines_per_sentence"])
        print(f'Cleaned dataset has {len(clean_dataset)} sentences')
        formated_dataset = list(map(self.toolset.get_formated_data, clean_dataset))
        print(f'Formated dataset has {len(formated_dataset)} sentences')
        self.write_result(formated_dataset)
        self.emite_log(f'task has finnished')

    def write_result(self, result):
        with self.output().open('w') as out_file:
            out_file.write('\n'.join(list(filter(lambda line: line is not None, result))))

    def emite_log(self, message):
        formated_datetime = datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        log.info(f'{formated_datetime}: {message}')

    def output(self):
        return luigi.LocalTarget(f'../outputs/cleaned_dataset_{self.dataset["name"]}')


class FormatDbPediaDataset(luigi.Task):
    path = luigi.Parameter(default='')

    # todo: Checagem de tipo de parametro
    def requires(self):
        dataset = {
            'name': f'dbpedia_relations_cleaned',
            'type': 'dbpedia',
            'path': self.path,
            'lines_per_sentence': 11
        }
        return FormatDataset(dataset=dataset)

    def output(self):
        return self.input()


if __name__ == '__main__':
    luigi.run()
