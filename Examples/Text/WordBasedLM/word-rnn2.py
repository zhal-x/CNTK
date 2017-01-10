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
from data_reader import *
from math import log

# model hyperparameters
hidden_dim = 256
num_layers = 1
minibatch_size = 257 # also how much time we unroll the RNN for

# model subgraph computing cross-entropy with sampled softmax
def cross_entropy_with_sampled_softmax(
    hidden_vector,           # Node providing the output of the recurrent layers
    target_vector,           # Node providing the expected labels (as sparse vectors)
    num_samples,             # Number of samples to use for sampled softmax
    sampling_weights,        # Node providing weights to be used for the weighted sampling
    vocab_dim,               # Vocabulary size
    hidden_dim,              # Row-dimension of the hidden vector
    allow_duplicates = False # Boolean flag to control wheather to use sampling with replacemement (allow_duplicates == True) or without replacement.
    ):
    bias = C.Parameter(shape = (vocab_dim, 1), init = C.init_bias_default_or_0, name='B')
    weights = C.Parameter(shape = (vocab_dim, hidden_dim), init = C.init_default_or_glorot_uniform, name='E')
    sample_selector = C.random_sample(sampling_weights, num_samples, allow_duplicates) # sparse matrix [num_samples * vocab_size]
    inclusion_probs = C.random_sample_inclusion_frequency(sampling_weights, num_samples, allow_duplicates) # dense row [1 * vocab_size]
    log_prior = C.log(inclusion_probs) # dense row [1 * vocab_dim]

    wS = C.times(sample_selector, weights, name='wS') # [num_samples * hidden_dim]
    zS = C.times_transpose(wS, hidden_vector, name='zS1') + C.times(sample_selector, bias, name='zS2') - C.times_transpose (sample_selector, log_prior, name='zS3')# [num_samples]

    # Getting the weight vector for the true label. Dimension hidden_dim
    wT = C.times(target_vector, weights, name='wT') # [1 * hidden_dim]
    zT = C.times_transpose(wT, hidden_vector, name='zT1') + C.times(target_vector, bias, name='zT2') - C.times_transpose(target_vector, log_prior, name='zT3') # [1]

    zSReduced = C.reduce_log_sum(zS)

    # Compute the cross entropy that is used for training.
    cross_entropy_on_samples = zSReduced - zT

    # For applying the model we also output a node providing the input for the full softmax
    z = C.times_transpose(weights, hidden_vector) + bias
    z = C.reshape(z, shape = (vocab_dim))
    zReduced = C.reduce_log_sum(zS)
    cross_entropy_with_full_softmax = zReduced - hidden_vector

    zSMax = C.reduce_max(zS)
    error_on_samples = C.less(zT, zSMax)
    return (z, cross_entropy_on_samples, error_on_samples, cross_entropy_with_full_softmax)

# Computes exp(z[index])/np.sum(exp[z]) for a one-dimensional numpy array in an numerically stable way.
def softmax(z, index):
    max_z = np.max(z)
    return np.exp(z[index] - max_z)/np.sum(np.exp(z - max_z))

# Computes the average cross entropy (in nats) for the specified text
def compute_average_cross_entropy(
    model_node,          # node computing the inputs to softmax
    word_ids,            # Sequence for which to compute the cross-entropy. Sequence is specified as a list of (word) ids.
    index_of_prime_word, # Index (of word) to prime the model
    vocab_dim):
    priming_input = C.one_hot([[index_of_prime_word]], vocab_dim)
    arguments = (priming_input, [True])

    total_cross_entropy = 0.0
    for word_id in word_ids:
        z = model_node.eval(arguments).flatten()
        p = softmax(z, word_id)
        total_cross_entropy += -log(p)
        x = C.one_hot([[int(word_id)]], vocab_dim)
        arguments = (x, [False])

    return total_cross_entropy / len(word_ids)


# Generate sequences from the model.
def sample(
    model_node,       # node computing the inputs to softmax
    ix_to_word,       # Dictionary mapping indices to tokens
    word_to_ix,       # Dictionary mapping tokens to indices
    vocab_dim,        # size of vocabulary
    prime_text='',    # text for priming the model
    use_hardmax=True, # Tells wehter hardmax should be used
    length=100,       # Length of sequence to generate
    alpha=1.0         # scaling exponent used for tweaking the probablities coming from the model. Used to control diversity of generated sequences.
    ):

    def sample_word(p):
        if use_hardmax:
            w = np.argmax(p, axis=2)[0,0]
        else:
            # normalize probabilities then take weighted sample
            p = np.exp(alpha* (p-np.max(p)))
            p = p/np.sum(p)
            w = np.random.choice(range(vocab_dim), p=p.ravel())
        return w

    plen = 1
    prime = -1

    # start sequence with first input    
    if prime_text != '':
        words = prime_text.split()
        plen = len(words)
        prime = word_to_ix[words[0]]
    else:
        prime = np.random.choice(range(vocab_dim))
    x = C.one_hot([[int(prime)]], vocab_dim)

    arguments = (x, [True])

    # setup a list for the output characters and add the initial prime text
    output = []
    output.append(prime)
    
    # loop through prime text
    for i in range(plen):
        p = model_node.eval(arguments)
        # reset
        if i < plen-1:
            idx = word_to_ix[words[i+1]]
        else:
            idx = sample_word(p)

        output.append(idx)
        x = C.one_hot([[int(idx)]], vocab_dim)

        arguments = (x, [False])
    
    # loop through length of generated text, sampling along the way
    for i in range(length-plen):
        p = model_node.eval(arguments)
        idx = sample_word(p)
        output.append(idx)
        x = C.one_hot([[int(idx)]], vocab_dim)

        arguments = (x, [False])

    # return output
    return ' '.join([ix_to_word[id] for id in output])


# Define the model to train
def create_model():
    
    return Sequential([        
        C.Embedding(hidden_dim),
        LayerStack(num_layers, lambda: 
                   Sequential([Stabilizer(), Recurrence(LSTM(hidden_dim), go_backwards=False)]))
    ])

# Model inputs
def create_inputs(vocab_dim):
    batch_axis = Axis.default_batch_axis()
    input_seq_axis = Axis('inputAxis')

    input_dynamic_axes = [batch_axis, input_seq_axis]
    input_sequence = input_variable(shape=vocab_dim, dynamic_axes=input_dynamic_axes, is_sparse = True)
    label_sequence = input_variable(shape=vocab_dim, dynamic_axes=input_dynamic_axes, is_sparse = True)
    
    return input_sequence, label_sequence

# Creates and trains a character-level language model
def train_lm(training_file, word_to_ix_file_path, sampling_weights_file_path, total_num_epochs, softmax_sample_size, alpha, input_test_text_file):

    # load the data and vocab
    data, word_to_ix, ix_to_word, data_size, vocab_dim = load_data_and_vocab(training_file, word_to_ix_file_path)
    
    # load the test data
    word_ids_test = text_file_to_word_ids(input_test_text_file, word_to_ix)

    # Model the source and target inputs to the model
    input_sequence, label_sequence = create_inputs(vocab_dim)

    # create the model
    rnn = create_model()
    
    # and apply it to the input sequence
    z = rnn(input_sequence)

    # setup the criterions (loss and metric)
    weights = load_sampling_weights(sampling_weights_file_path)
    smoothed_weights = np.float32( np.power(weights, alpha))
    sampling_weights = C.reshape(C.Constant(smoothed_weights), shape = (1,vocab_dim))
    model, ce, error_on_samples, cross_entropy_with_full_softmax = cross_entropy_with_sampled_softmax(z, label_sequence, softmax_sample_size, sampling_weights, vocab_dim, hidden_dim)

    # Instantiate the trainer object to drive the model training
    lr_per_sample = learning_rate_schedule(0.01, UnitType.sample)
    momentum_time_constant = momentum_as_time_constant_schedule(1100)
    clipping_threshold_per_sample = 5.0
    gradient_clipping_with_truncation = True
    learner = momentum_sgd(ce.parameters, lr_per_sample, momentum_time_constant, 
                           gradient_clipping_threshold_per_sample=clipping_threshold_per_sample,
                           gradient_clipping_with_truncation=gradient_clipping_with_truncation)
    trainer = Trainer(model, ce, error_on_samples, learner)

    minibatches_per_epoch = int((data_size / minibatch_size))
    total_num_minibatches = total_num_epochs * minibatches_per_epoch
    
    # print out some useful training information
    log_number_of_parameters(z) ; print()
    log_number_of_parameters(model) ; print()
    progress_printer = ProgressPrinter(freq=1, tag='Training')    
    
    epoche_count = 0
    num_trained_samples_within_current_epoche = 0
    num_trained_samples_since_last_report = 0

    for i in range(0, total_num_minibatches):

        if num_trained_samples_within_current_epoche + minibatch_size+1 >= data_size:
            num_trained_samples_within_current_epoche = 0
            epoche_count += 1
            model_filename = "models/lm_epoch%d.dnn" % epoche_count
            model.save_model(model_filename)
            print("Saved model to '%s'" % model_filename)

        # get the datafor next batch
        features, labels = get_data(num_trained_samples_within_current_epoche, minibatch_size, data, word_to_ix, vocab_dim)

        # Specify the mapping of input variables in the model to actual minibatch data to be trained with
        # If it's the start of the data, we specify that we are looking at a new sequence (True)
        mask = [False] 
        if num_trained_samples_within_current_epoche == 0:
            mask = [True]
        arguments = ({input_sequence : features, label_sequence : labels}, mask)
        trainer.train_minibatch(arguments)

        
        num_samples_between_progress_report = 1000
        if num_trained_samples_since_last_report >= num_samples_between_progress_report:
            progress_printer.update_with_trainer(trainer, with_metric=True) # log progress
            print(sample(model, ix_to_word, word_to_ix, vocab_dim))
            print("start....")
            average_cross_entropy = compute_average_cross_entropy(model, word_ids_test, 1, vocab_dim)
            print("average cross entropy:" + str(average_cross_entropy))
            num_samples_between_progress_report = 0

        num_trained_samples_since_last_report += minibatch_size
        num_trained_samples_within_current_epoche += minibatch_size

def load_and_sample(model_filename, word_to_ix_file_path, prime_text='', use_hardmax=False, length=1000, alpha=1.0):
    
    # load the model
    model = load_model(model_filename)
    
    # load the vocab
    word_to_ix, ix_to_word = load_word_to_ix(word_to_ix_file_path)
    
    output = sample(model, ix_to_word, word_to_ix, len(word_to_ix), prime_text=prime_text, use_hardmax=use_hardmax, length=length, alpha=alpha)
    
    ff = open('output.txt', 'w', encoding='utf-8')
    ff.write(output)
    ff.close()

if __name__=='__main__':
    print("press return")
    input()
    print("continuing...")

    num_epochs = 32
    input_training_text_file = "ptbData/ptb.train.txt"
    input_test_text_file = "ptbData/ptb.test.subset.txt"
    input_word2index_file = "ptbData/ptb.word2id.txt"
    input_sampling_weights = "ptbData/ptb.freq.txt"
    

    # train the LM    
    train_lm(input_training_text_file, input_word2index_file, input_sampling_weights, num_epochs, 1000, 0.5, input_test_text_file)

    # load and sample
    priming_text = ""
    final_model_file = "models/lm_epoch%i.dnn" % (num_epochs-1)
    load_and_sample(final_model_file, input_word2index_file, prime_text = priming_text, use_hardmax = False, length = 100, alpha = 1.0)
