# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import cntk as C
import os

# Map word sequence to input and output sequence
def get_data(word_sequence, word_to_id, vocab_dim):
    xi = [word_to_id[word] for word in word_sequence[ : -1] ]
    yi = [word_to_id[word] for word in word_sequence[1 : ] ]
    
    X = C.one_hot([xi], vocab_dim)
    Y = C.one_hot([yi], vocab_dim)

    return X, Y, len(yi)

# read the mapping word_to_id from file (tab sepparted)
def load_word_to_id(word_to_id_file_path):
    word_to_id = {}
    id_to_word = {}
    f = open(word_to_id_file_path,'r')
    for line in f:
        entry = line.split('\t')
        if len(entry) == 2:
            word_to_id[entry[0]] = int(entry[1])
            id_to_word[int(entry[1])] = entry[0]

    return (word_to_id, id_to_word)

# reads a file with one number per line and returns the numbers as a list
def load_sampling_weights(sampling_weights_file_path):
    weights = []
    f = open(sampling_weights_file_path,'r')
    for line in f:
        if len(line) > 0:
            weights.append(float(line))
    return weights

# reads a text file, generates tokens, and maps and converts them into a list of word-ids,
def text_file_to_word_ids(text_file_path, word_to_id):
    rel_path = text_file_path
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)
    text_file = open(path, "r")
    word_sequence = []
    for line in text_file:
        words = line.split()
        for word in words:
            if not word in word_to_id:
                print ("ERROR: while reading file '" + text_file_path + "' word without id: " + word)
                sys.exit()
            word_sequence.append(word_to_id[word])

    return word_sequence

# Read a text and convert the tokens to a corresponding list of ids using the providided token-to-id map
def load_data_and_vocab(
    text_file_path,      # input text file
    word_to_id_file_path # Dictionary mapping tokens to ids
    ):
    word_to_id, id_to_word = load_word_to_id(word_to_id_file_path)

    # represent text be sequence of words indices 'word_sequence'
    rel_path = text_file_path
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)
    text_file = open(path, "r")
    word_sequences = []
    for line in text_file:
        word_sequence = []
        words = line.split()
        for word in words:
            if not word in word_to_id:
                print ("ERROR: word without id: " + word)
                sys.exit()
            word_sequence.append(word)
        word_sequences.append(word_sequence)

    word_count = len(word_sequence)
    vocab_size = len(word_to_id)

    return word_sequences, word_to_id, id_to_word, vocab_size
