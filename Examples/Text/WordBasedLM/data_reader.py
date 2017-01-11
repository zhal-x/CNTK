import cntk as C
import os

# Get data
def get_data(p, minibatch_size, data, word_to_ix, vocab_dim):

    # the word LM predicts the next character so get sequences offset by 1
    xi = [word_to_ix[ch] for ch in data[p:p+minibatch_size]]
    yi = [word_to_ix[ch] for ch in data[p+1:p+minibatch_size+1]]
    
    X = C.one_hot([xi], vocab_dim)
    Y = C.one_hot([yi], vocab_dim)

    # return a list of numpy arrays for each of X (features) and Y (labels)
    return X, Y

# read the mapping word_to_ix from file (tab sepparted)
def load_word_to_ix(word_to_ix_file_path):
    word_to_ix = {}
    ix_to_word = {}
    f = open(word_to_ix_file_path,'r')
    for line in f:
        entry = line.split('\t')
        if len(entry) == 2:
            word_to_ix[entry[0]] = int(entry[1])
            ix_to_word[int(entry[1])] = entry[0]

    return (word_to_ix, ix_to_word)

# reads a file with one number per line and returns the numbers as a list
def load_sampling_weights(sampling_weights_file_path):
    weights = []
    f = open(sampling_weights_file_path,'r')
    for line in f:
        if len(line) > 0:
            weights.append(float(line))
    return weights

# reads a text file, generates tokens, and maps and converts them into a list of word-ids,
def text_file_to_word_ids(text_file_path, word_to_ix):
    rel_path = text_file_path
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)
    text_file = open(path, "r")
    word_sequence = []
    for line in text_file:
        words = line.split()
        for word in words:
            if not word in word_to_ix:
                print ("ERROR: while reading file '" + text_file_path + "' word without id: " + word)
                sys.exit()
            word_sequence.append(word_to_ix[word])

    return word_sequence

# read text and map it into a list of word indices
def load_data_and_vocab(text_file_path, word_to_ix_file_path):
    word_to_ix, ix_to_word = load_word_to_ix(word_to_ix_file_path)

    # represent text be sequence of words indices 'word_sequence'
    rel_path = text_file_path
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)
    text_file = open(path, "r")
    word_sequence = []
    for line in text_file:
        words = line.split()
        for word in words:
            if not word in word_to_ix:
                print ("ERROR: word without id: " + word)
                sys.exit()
            word_sequence.append(word)

    word_count = len(word_sequence)
    vocab_size = len(word_to_ix)

    return word_sequence, word_to_ix, ix_to_word, word_count, vocab_size
