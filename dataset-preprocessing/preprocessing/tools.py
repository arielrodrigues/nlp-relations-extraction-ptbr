# -*- coding: utf-8 -*-
import re
import json

"""
Module Docstring
"""

__author__ = "Ariel Rodrigues"
__version__ = "0.1.0"
__license__ = ""


def is_trash(line):
    return '*****' in line or not line.strip()


def remove_unwanted_characters(line, characters=[]):
    for char in characters:
        line = line.replace(char, '')
    return line


def extract_key_and_value(line):
    key, value = line.split(' : ')
    return key, value


def tokenize(sentence):
    return re.findall(r"[\w']+", sentence)


def get_lines_from_file(file):
    with open(file, 'r') as infile:
        lines = infile.readlines()
        infile.close()
    return lines


def find_entity_position_in_tokens(entity, tokens):
    return tokens.index(entity) if (entity in tokens) else None


def entity_in_tokens(entity, tokens):
    return entity in tokens


def find_entity_on_tokens(entity, tokens):
    if not entity_in_tokens(entity, tokens):
        first_token_entity = tokenize(entity)[0]
        if not entity_in_tokens(first_token_entity, tokens):
            return None
        return find_entity_position_in_tokens(first_token_entity, tokens)
    return find_entity_position_in_tokens(entity, tokens)


def extract_entities_positions(sentence, entity1, entity2):
    tokens = tokenize(sentence)

    pos_ent1 = find_entity_on_tokens(entity1, tokens)
    pos_ent2 = find_entity_on_tokens(entity2, tokens)

    return f'{pos_ent1}\t{pos_ent2}' if (pos_ent1 is not None and pos_ent2 is not None) else None


def get_formated_data(data):
    pos = extract_entities_positions(data["SENTENCE"], data["ENTITY1"], data["ENTITY2"])
    rel_type = f'{data["REL TYPE"]}'
    sentence = data["SENTENCE"]
    return f'{rel_type}\t{pos}\t{sentence}' if pos else None


def clean_dataset(raw_lines, lines_per_sentence):
    count = -1
    number_of_sentences = len(raw_lines) / lines_per_sentence
    sentences = [dict() for i in range(int(number_of_sentences))]
    for line in raw_lines:
        if is_trash(line):
            continue
        clean_line = remove_unwanted_characters(line, ['\n'])
        key, value = extract_key_and_value(clean_line)

        if key == 'MANUALLY CHECKED':
            continue
        elif key == 'SENTENCE':
            count += 1
            sentences[count][key] = value
        else:
            sentences[count][key] = value
    return sentences


def print_number_of_sentences_by_classes(sentences):
    counters = dict()
    for sentence in sentences:
        if sentence is None: continue
        label = sentence.strip().split('\t')[0]
        if label in counters.keys(): counters[label] += 1
        else: counters[label] = 1
    print('SENTENCES BY CLASSES:')
    print(json.dumps(counters, indent=2, sort_keys=False))