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

# model hyperparameters
hidden_dim = 256
num_layers = 1
minibatch_size = 257 # also how much time we unroll the RNN for

def cross_entropy_with_sampled_softmax(output_vector, target_vector, num_samples, sampling_weights, vocab_dim, hidden_dim, allow_duplicates=False, name=''):
    bias = C.Parameter(shape = (vocab_dim, 1), init = C.init_bias_default_or_0, name='B')
    weights = C.Parameter(shape = (vocab_dim, hidden_dim), init = C.init_default_or_glorot_uniform, name='E')
    sample_selector = C.random_sample(sampling_weights, num_samples, allow_duplicates) # sparse matrix [num_samples * vocab_size]
    inclusion_probs = C.random_sample_inclusion_frequency(sampling_weights, num_samples, allow_duplicates) # dense row [1 * vocal_size]
    log_prior = C.log(inclusion_probs) # dense row [1 * vocal_size]

    wS = C.times(sample_selector, weights, name='wS') # [num_samples * hidden_dim] 
    zS = C.times_transpose(wS, output_vector, name='zS1') + C.times(sample_selector, bias, name='zS2') - C.times_transpose (sample_selector, log_prior, name='zS3')# [numSamples]

    # Getting the weight vector for the true label. Dimension numHidden
    wT = C.times(target_vector, weights, name='wT') # [1 * numHidden]
    zT = C.times_transpose(wT, output_vector, name='zT1') + C.times(target_vector, bias, name='zT2') - C.times_transpose(target_vector, log_prior, name='zT3') # [1]

    zSReduced = C.reduce_log_sum(zS)

    cross_entropy_on_samples = zSReduced - zT

    # for testing purposes setup cross entropy with full softmax
    z = C.times_transpose(weights, output_vector) + bias
    z = C.reshape(z, shape = (vocab_dim))
    zReduced = C.reduce_log_sum(zS)
    cross_entropy_with_full_softmax = zReduced - output_vector

    zSMax = C.reduce_max(zS)
    error_on_samples = C.less(zT, zSMax)
    return (z, cross_entropy_on_samples, error_on_samples, cross_entropy_with_full_softmax)

# Generate sequences from the model
def sample(root, ix_to_word, vocab_dim, word_to_ix, prime_text='', use_hardmax=True, length=100, alpha=1.0):

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
        p = root.eval(arguments)
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
        p = root.eval(arguments)
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
def train_lm(training_file, word_to_ix_file_path, sampling_weights_file_path, total_num_epochs, softmax_sample_size, alpha):

    # load the data and vocab
    data, word_to_ix, ix_to_char, data_size, vocab_dim = load_data_and_vocab(training_file, word_to_ix_file_path)

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
    model, ce, error_on_samples, cross_entropy_with_full_softmax = cross_entropy_with_sampled_softmax(z, label_sequence, softmax_sample_size, sampling_weights, vocab_dim, hidden_dim, name = 'sampled_softmax')

    # Instantiate the trainer object to drive the model training
    lr_per_sample = learning_rate_schedule(0.001, UnitType.sample)
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
    p = 0
    for i in range(0, total_num_minibatches):

        if p + minibatch_size+1 >= data_size:
            p = 0
            epoche_count += 1
            model_filename = "models/lm_epoch%d.dnn" % epoche_count
            model.save_model(model_filename)
            print("Saved model to '%s'" % model_filename)

        # get the datafor next batch
        features, labels = get_data(p, minibatch_size, data, word_to_ix, vocab_dim)

        # Specify the mapping of input variables in the model to actual minibatch data to be trained with
        # If it's the start of the data, we specify that we are looking at a new sequence (True)
        mask = [False] 
        if p == 0:
            mask = [True]
        arguments = ({input_sequence : features, label_sequence : labels}, mask)
        trainer.train_minibatch(arguments)

        progress_printer.update_with_trainer(trainer, with_metric=True) # log progress
        
        num_minbatches_between_printing_samples = 2
        if i % num_minbatches_between_printing_samples == 0:
            print(sample(model, ix_to_char, vocab_dim, word_to_ix))

        p += minibatch_size
        

def load_and_sample(model_filename, word_to_ix_file_path, prime_text='', use_hardmax=False, length=1000, alpha=1.0):
    
    # load the model
    model = load_model(model_filename)
    
    # load the vocab
    word_to_ix, ix_to_char = load_word_to_ix(word_to_ix_file_path)
    
    output = sample(model, ix_to_char, len(word_to_ix), word_to_ix, prime_text=prime_text, use_hardmax=use_hardmax, length=length, alpha=alpha)
    
    ff = open('output.txt', 'w', encoding='utf-8')
    ff.write(output)
    ff.close()

if __name__=='__main__':
    print("press return")
    input()
    print("continuing...")

    num_epochs = 32
    input_text_file = "ptbData/ptb.train.txt"
    input_word2index_file = "ptbData/ptb.word2id.txt"
    input_sampling_weights = "ptbData/ptb.freq.txt"
    

    # train the LM    
    train_lm(input_text_file, input_word2index_file, input_sampling_weights, num_epochs, 1000, 0.5)

    # load and sample
    priming_text = ""
    final_model_file = "models/lm_epoch%i.dnn" % (num_epochs-1)
    load_and_sample(final_model_file, input_word2index_file, prime_text = priming_text, use_hardmax = False, length = 100, alpha = 1.0)
