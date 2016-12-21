# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import cntk as C
from cntk import Trainer, Axis
from cntk.learner import momentum_sgd, momentum_as_time_constant_schedule, learning_rate_schedule, UnitType
from cntk.ops import input_variable, cross_entropy_with_softmax, classification_error
from cntk.ops.functions import load_model
from cntk.blocks import LSTM, Stabilizer
from cntk.layers import Recurrence, Dense
from cntk.models import LayerStack, Sequential
from cntk.utils import log_number_of_parameters, ProgressPrinter

# model hyperparameters
hidden_dim = 32
num_layers = 1
minibatch_size = 100 # also how much time we unroll the RNN for

def cross_entropy_with_sampled_softmax(output_vector, target_vector, num_samples, sampling_weights, vocab_dim, hidden_dim, allow_duplicates=False, name=''):
    bias = C.Parameter(shape = (vocab_dim, 1), init = C.init_bias_default_or_0, name='B')
    weights = C.Parameter(shape = (vocab_dim, hidden_dim), init = C.init_default_or_glorot_uniform, name='E')
    sample_selector = C.random_sample(sampling_weights, num_samples, allow_duplicates) # sparse matrix [num_samples * vocab_size]
    inclusion_probs = C.random_sample_inclusion_frequency(sampling_weights, num_samples, allow_duplicates) # dense row [1 * vocal_size]
    log_prior = C.log(inclusion_probs) # dense row [1 * vocal_size]
    print("shape weights %s" % str(weights.shape))
    print("shape logPrior %s" % str(log_prior.shape))

    wS = C.times(sample_selector, weights) # [num_samples * hidden_dim] 
    print("shape ws %s" % str(wS.shape))
    zS = C.times_transpose(wS, output_vector) + C.times(sample_selector, bias) - C.times_transpose (sample_selector, log_prior)# [numSamples]
    print("shape zs %s" % str(zS.shape))

    # Getting the weight vector for the true label. Dimension numHidden
    wT = C.times(target_vector, weights) # [1 * numHidden]
    print("shape wT %s" % str(wT.shape))
    zT = C.times_transpose(wT, output_vector) +  C.times(target_vector, bias) - C.times_transpose(target_vector, log_prior) # [1]
    print("shape zT %s" % str(zT.shape))

    zSReduced = C.reduce_log_sum(zS)
    print("shape zSReduced %s" % str(zSReduced.shape))
                                        
    # The label (true class), might already be among the sampled classes.
    # To get the 'partition function' over the union of label and sampled classes
    # we need to LogPlus zT if the label is not among the sampled classes.
    labelIsInSampled = C.reduce_sum (C.times_transpose(sample_selector, target_vector))

    return (C.times_transpose(weights, output_vector), zSReduced - zT)

    # Check whether the label would have been predicted correctly (among the set of random samples plus label)
    # error = 1 if the label would have been predicted wrongly
    # zSMax = ReduceMax (zS)
    # error = Less (zT, zSMax)



# Get data
def get_data(p, minibatch_size, data, word_to_ix, vocab_dim):

    # the word LM predicts the next character so get sequences offset by 1
    xi = [word_to_ix[ch] for ch in data[p:p+minibatch_size]]
    yi = [word_to_ix[ch] for ch in data[p+1:p+minibatch_size+1]]
    
    X = C.one_hot([xi], vocab_dim)
    Y = C.one_hot([yi], vocab_dim)

    # return a list of numpy arrays for each of X (features) and Y (labels)
    return X, Y

# Sample words from the model
def sample(root, ix_to_word, vocab_dim, word_to_ix, prime_text='', use_hardmax=True, length=100, alpha=1.0):

    # alpha < 1 means smoother; alpha == 1.0 means same; alpha > 1 means more peaked
    def apply_temp(p):
        # apply temperature
        p = np.power(p, alpha)
        # normalize and return
        return (p / np.sum(p))

    def sample_word(p):
        if use_hardmax:
            w = np.argmax(p, axis=2)[0,0]
        else:
            # normalize probabilities then take weighted sample
            p = np.exp(p)
            p = apply_temp(p)
            w = np.random.choice(range(vocab_dim), p=p.ravel())
        return w

    plen = 1
    prime = -1

    # start sequence with first input    
    x = np.zeros((1, vocab_dim), dtype=np.float32)    
    if prime_text != '':
        words = prime_text.split()
        plen = len(words)
        prime = word_to_ix[words[0]]
    else:
        prime = np.random.choice(range(vocab_dim))
    x[0, prime] = 1
    arguments = ([x], [True])

    # setup a list for the output characters and add the initial prime text
    output = []
    output.append(prime)
    
    # loop through prime text
    for i in range(plen):            
        p = root.eval(arguments)        
        
        # reset
        x = np.zeros((1, vocab_dim), dtype=np.float32)
        if i < plen-1:
            idx = word_to_ix[words[i+1]]
        else:
            idx = sample_word(p)

        output.append(idx)
        x[0, idx] = 1            
        arguments = ([x], [False])
    
    # loop through length of generated text, sampling along the way
    for i in range(length-plen):
        p = root.eval(arguments)
        idx = sample_word(p)
        output.append(idx)

        x = np.zeros((1, vocab_dim), dtype=np.float32)
        x[0, idx] = 1
        arguments = ([x], [False])

    # return output
    return ' '.join([ix_to_word[id] for id in output])

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

# read text and map it into a list of word indices
def load_data_and_vocab(training_file_path, word_to_ix_file_path):
    word_to_ix, ix_to_word = load_word_to_ix(word_to_ix_file_path)

    # represent text be sequence of words indices 'word_sequence'
    rel_path = training_file_path
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

# Creates the model to train
def create_model(output_dim):
    
    return Sequential([        
        LayerStack(num_layers, lambda: 
                   Sequential([Stabilizer(), Recurrence(LSTM(hidden_dim), go_backwards=False)]))
    ])

# Model inputs
def create_inputs(vocab_dim):
    batch_axis = Axis.default_batch_axis()
    input_seq_axis = Axis('inputAxis')

    input_dynamic_axes = [batch_axis, input_seq_axis]
    input_sequence = input_variable(shape=vocab_dim, dynamic_axes=input_dynamic_axes)
    label_sequence = input_variable(shape=vocab_dim, dynamic_axes=input_dynamic_axes)
    
    return input_sequence, label_sequence

# Creates and trains a character-level language model
def train_lm(training_file, word_to_ix_file_path, total_num_epochs):

    # load the data and vocab
    data, word_to_ix, ix_to_char, data_size, vocab_dim = load_data_and_vocab(training_file, word_to_ix_file_path)

    # Model the source and target inputs to the model
    input_sequence, label_sequence = create_inputs(vocab_dim)

    # create the model
    rnn = create_model(vocab_dim)
    
    # and apply it to the input sequence
    z = rnn(input_sequence)

    # setup the criterions (loss and metric)
    sampling_weights = C.reshape(C.Constant(np.array([1,1,1,1,1])), shape = (1,vocab_dim))
    model, ce = cross_entropy_with_sampled_softmax(z, label_sequence, 2, sampling_weights, vocab_dim, hidden_dim, name = 'sampled_softmax')

    errs = C.constant(0) + 0

    # Instantiate the trainer object to drive the model training
    lr_per_sample = learning_rate_schedule(0.001, UnitType.sample)
    momentum_time_constant = momentum_as_time_constant_schedule(1100)
    clipping_threshold_per_sample = 5.0
    gradient_clipping_with_truncation = True
    learner = momentum_sgd(ce.parameters, lr_per_sample, momentum_time_constant, 
                           gradient_clipping_threshold_per_sample=clipping_threshold_per_sample,
                           gradient_clipping_with_truncation=gradient_clipping_with_truncation)
    trainer = Trainer(model, ce, errs, learner)

    epochs = 50
    minibatches_per_epoch = int((data_size / minibatch_size))
    total_num_minibatches = total_num_epochs * minibatches_per_epoch
    
    # print out some useful training information
    log_number_of_parameters(z) ; print()
    progress_printer = ProgressPrinter(freq=100, tag='Training')    
    
    epoche_count = 0
    p = 0
    for i in range(0, total_num_minibatches):

        if p + minibatch_size+1 >= data_size:
            p = 0
            epoche_count += 1
            model_filename = "models/lm_epoch%d.dnn" % epoche_count
            z.save_model(model_filename)
            print("Saved model to '%s'" % model_filename)

        # get the data            
        features, labels = get_data(p, minibatch_size, data, word_to_ix, vocab_dim)

        # Specify the mapping of input variables in the model to actual minibatch data to be trained with
        # If it's the start of the data, we specify that we are looking at a new sequence (True)
        mask = [False] 
        if p == 0:
            mask = [True]
        arguments = ({input_sequence : features, label_sequence : labels}, mask)
        trainer.train_minibatch(arguments)

        progress_printer.update_with_trainer(trainer, with_metric=True) # log progress
        
        num_minbatches_between_printing_samples = 1000
#        if i % num_minbatches_between_printing_samples == 0:
#            print(sample(z, ix_to_char, vocab_dim, word_to_ix))

        p += minibatch_size
        

def load_and_sample(model_filename, word_to_ix_file_path, prime_text='', use_hardmax=False, length=1000, temperature=1.0):
    
    # load the model
    model = load_model(model_filename)
    
    # load the vocab
    word_to_ix, ix_to_char = load_word_to_ix(word_to_ix_file_path)
    
    output = sample(model, ix_to_char, len(word_to_ix), word_to_ix, prime_text=prime_text, use_hardmax=use_hardmax, length=length, temperature=temperature)
    
    ff = open('output.txt', 'w', encoding='utf-8')
    ff.write(output)
    ff.close()

if __name__=='__main__':    

    # train the LM    
    train_lm("data/test.txt", "data/test_w2i.txt", 1)

    # load and sample
    #text = "aaa bbb ccc ddd eee aaa bbb ccc"
    #load_and_sample("models/lm_epoch49.dnn", "data/test_w2i.txt", prime_text=text, use_hardmax=False, length=100, temperature=1.0)
