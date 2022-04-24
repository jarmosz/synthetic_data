import random
import hunspell
import spacy
from tokenizer import Tokenizer
from scipy.stats import norm
import pandas as pd
import regex as re
import glob
import threading

# pip install hunspell spacy
# python -m spacy download pl_core_news_lg
# sed -ri '/^\s*$/d' ./datasets_original/oscar/oscar_pl_test.txt
#  unzip -p oscar_filtered.zip | sed -r '/^\s*$/d' | gawk 'NF>6'  > oscar_filtered.txt


class SyntheticErrorsGenerator:
    def __init__(self):
        self.substitution_prob = 0.7
        self.remain_prob = 0.3
        self.input_dataframe = pd.DataFrame([], columns=['sentence'])
        self.output_dataframe = pd.DataFrame([], columns=['sentence'])
        spacy.load('pl_core_news_lg')
        self.spellchecker = hunspell.HunSpell('./pl.dic', './pl.aff')
        self.tokenizer = Tokenizer()

    def read_input_file(self, input_filename):
        with open(input_filename, encoding="utf-8", mode='r') as input:
            yield from input.readlines()

    def delete_character(self, str, idx):
        return str[:idx] + str[idx+1:]

    def delete(self, tokens, idx):
        tokens.pop(idx)
        return tokens

    def swap_characters(self, str, idx):
        strlst = list(str)
        if not (len(str) - 1) == idx:
            strlst[idx], strlst[idx+1] = strlst[idx+1], strlst[idx]
            return "".join(strlst)
        else:
            strlst[idx-1], strlst[idx] = strlst[idx-1], strlst[idx]
            return "".join(strlst)

    def spelling_error(self, tokens):
        errors_matrix = {
            'ą': 'a',
            'ć': 'c',
            'ę': 'e',
            'ł': 'l',
            'ń': 'n',
            "ó": 'u',
            "u": 'ó',
            'ś': 's',
            'ź': 'z',
            'ż': 'z'
        }

        letters_existing_in_word = []
        for letter, _ in errors_matrix.items():
            if letter in tokens:
                letters_existing_in_word.append(letter)
        
        if len(letters_existing_in_word) > 0:
            letter_to_replace = random.choice(letters_existing_in_word)
            tokens = tokens.replace(letter_to_replace, errors_matrix[letter_to_replace])

        return tokens

    def swap(self, tokens, idx):
        if not (len(tokens) - 1) == idx:
            tokens[idx], tokens[idx + 1] = tokens[idx + 1], tokens[idx]
        else:
            tokens[idx - 1], tokens[idx] = tokens[idx], tokens[idx - 1]
        return tokens

    def add_random(self, tokens, idx):
        confusion_set = self.spellchecker.suggest(tokens[idx])[:3]
        if len(confusion_set) > 1:
            word_to_replace = random.sample(confusion_set, 1)[0]
            tokens.insert(idx + 1, word_to_replace)
        return tokens

    def add_random_character(self, str, idx):
        confusion_set = self.spellchecker.suggest(str[idx])[:3]
        if len(confusion_set) > 1:
            char_to_replace = random.sample(confusion_set, 1)[0].lower()
            return str[:idx] + char_to_replace + str[idx:]
        return str

    def substitute_delete_add(self, tokens, token_idx, operation):
        if operation == 'DELETE':
            return self.delete(tokens, token_idx)
        elif operation == 'SWAP':
            return self.swap(tokens, token_idx)
        elif operation == 'ADD_RANDOM':
            return self.add_random(tokens, token_idx)
        elif operation == 'SWAP_CHARACTERS':
            return self.swap_characters(tokens, token_idx)
        elif operation == 'DELETE_CHARACTER':
            return self.delete_character(tokens, token_idx)
        elif operation == 'ADD_RANDOM_CHARACTER':
            return self.add_random_character(tokens, token_idx)
        elif operation == 'SPELLING_ERROR':
            return self.spelling_error(tokens)
         
    def introduce_error(self, line):
        tokens = self.tokenizer.tokenize(line)
        num_words_to_change = round(abs(norm.mean(0.15, 0.2) * norm.std(0.15, 0.2)) * len(line))
        if num_words_to_change > len(set(tokens)):
            num_words_to_change = 1
        words_to_change = random.sample(set(tokens), num_words_to_change)
        num_words_to_change_letters = round(len(tokens) * 0.1)

        words_for_spelling_errors = [tokens.index(word) for word in tokens if word not in words_to_change]
        for idx in random.sample(words_for_spelling_errors, num_words_to_change_letters):
            word = tokens[idx]
            if word.isalnum():
                random_number = random.random()
                random_operation = random.sample(['DELETE_CHARACTER', 'SWAP_CHARACTERS', 'ADD_RANDOM_CHARACTER', 'SPELLING_ERROR'], 1)[0]
                random_idx = random.sample(range(0, len(word)), 1)[0]
                tokens[idx] = self.substitute_delete_add(word, random_idx, random_operation)

        for word_to_change in words_to_change:
            idx = tokens.index(word_to_change)
            random_number = random.random()
            random_operation = random.sample(['DELETE', 'SWAP', 'ADD_RANDOM'], 1)[0]
            if random_number < self.remain_prob:
                tokens = self.substitute_delete_add(tokens, idx, random_operation)
            elif random_number < self.substitution_prob:
                word_to_replace = ''
                confusion_set = self.spellchecker.suggest(word_to_change)[:3]
                if len(confusion_set) > 1:
                    word_to_replace = random.sample(confusion_set, 1)[0]
                    while(word_to_replace == word_to_change):
                        word_to_replace = random.sample(confusion_set, 1)[0]
                    tokens[idx] = word_to_replace
                else:
                    tokens = self.substitute_delete_add(tokens, idx, random_operation)
        return ' '.join(tokens)

    def generate_synthetic_errors_from_folder(self, folder_path):
        for idx, path in enumerate(glob.glob(folder_path)[:11]):
            t = threading.Thread(target=self.generate_synthetic_errors_from_file, args=(path, f'./datasets_original/oscar/splitted_oscar/input{idx}.txt', f'./datasets_original/oscar/splitted_oscar/output{idx}.txt'))
            t.start()


    def generate_synthetic_errors_from_file(self, source_filename, input_filename, output_filename):
        with open(input_filename, encoding="utf-8", mode="w") as input:
            with open(output_filename, encoding="utf-8", mode="w") as output:
                for line in self.read_input_file(source_filename):
                    if len(line.split()) > 7:
                        new_line = line.strip()
                        new_line = new_line[0].capitalize() + new_line[1:]
                        new_line_with_error = self.introduce_error(new_line)
                        input.write(new_line + "\n")
                        output.write(new_line_with_error + "\n")


synthetic_errors_generator = SyntheticErrorsGenerator()
synthetic_errors_generator.generate_synthetic_errors_from_file('./datasets_original/oscar/splitted_oscar/output_fileaa', './datasets_original/oscar/splitted_oscar/results/input1.txt', './datasets_original/oscar/splitted_oscar/results/output1.txt')

# synthetic_errors_generator.generate_synthetic_errors_from_folder(folder_path='./datasets_original/oscar/splitted_oscar/*')
# synthetic_errors_generator.generate_synthetic_errors('./datasets_original/oscar/splitted_oscar/output_fileaa', './datasets_original/oscar/splitted_oscar/input1.txt', './datasets_original/oscar/splitted_oscar/output1.txt')
# synthetic_errors_generator.generate_synthetic_errors('./datasets_original/lektury/input2_lektury.txt', './datasets_original/lektury/input2.txt', './datasets_original/lektury/output2.txt')

