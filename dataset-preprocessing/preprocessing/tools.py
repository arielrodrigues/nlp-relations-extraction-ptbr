# -*- coding: utf-8 -*-
import spacy

spacy_nlp = spacy.load('pt_core_news_sm')

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


def extract_tokens_and_entities(sentence):
    parsed_sentence = spacy_nlp(sentence)
    tokens = [token.text for token in parsed_sentence]
    entities = parsed_sentence.ents
    return tokens, entities


def get_lines_from_file(file):
    with open(file, 'r') as infile:
        lines = infile.readlines()
        infile.close()
    return lines


def find_entity_position_in_tokens(entity, tokens):
    return tokens.index(entity) if (entity in tokens) else None


def remove_duplicates_in_list(raw_list):
    return sorted(set(raw_list), key=raw_list.index)


def tokenizer(sentence):
    tokens, entities = extract_tokens_and_entities(sentence)
    tokens_with_entities = tokens[:]
    for entity in entities:
        for index, token in enumerate(tokens):
            if token in entity.text:
                tokens_with_entities[index] = entity.text

    return remove_duplicates_in_list(tokens_with_entities)


def entities_in_tokens(entity1, entity2, tokens):
    return entity1 in tokens and entity2 in tokens


def extract_entities_positions(sentence, entity1, entity2):
    tokens = tokenizer(sentence)
    pos_ent1, pos_ent2 = None, None

    if entities_in_tokens(entity1, entity2, tokens):
        pos_ent1 = find_entity_position_in_tokens(entity1, tokens)
        pos_ent2 = find_entity_position_in_tokens(entity2, tokens)
    else:
        first_token_entity1 = entity1.split()[0]
        first_token_entity2 = entity2.split()[0]
        if entities_in_tokens(first_token_entity1, first_token_entity2, tokens):
            pos_ent1 = find_entity_position_in_tokens(first_token_entity1, tokens)
            pos_ent2 = find_entity_position_in_tokens(first_token_entity2, tokens)

    return f'{pos_ent1}\t{pos_ent2}' if (pos_ent1 and pos_ent2) else None


def format_rel_type(rel_type):
    return f'{rel_type}(e1,e2)'


def get_formated_data(data):
    pos = extract_entities_positions(data["SENTENCE"], data["ENTITY1"], data["ENTITY2"])
    rel_type = format_rel_type(data["REL TYPE"])
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
