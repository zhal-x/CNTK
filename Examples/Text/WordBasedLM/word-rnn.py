# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import cntk as C
import timeit
from cntk import Trainer, Axis
from cntk.learner import momentum_sgd, momentum_as_time_constant_schedule, learning_rate_schedule, UnitType
from cntk.ops import input_variable, cross_entropy_with_softmax, classification_error
from cntk.ops.functions import load_model
from cntk.blocks import LSTM, Stabilizer
from cntk.layers import Recurrence, Dense
from cntk.models import LayerStack, Sequential
from cntk.utils import log_number_of_parameters, ProgressPrinter
from data_reader import *
from math import log, exp
from download_data import Paths

from cntk.device import set_default_device, cpu, gpu

# Setting global parameters
# Model sizes similar to https://arxiv.org/pdf/1409.2329.pdf and https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py
type = 'medium'

if type == 'medium':
    hidden_dim = 650
    num_layers = 2
    num_epochs = 39
    alpha = 0.75
    learning_rate = 0.003
    softmax_sample_size = 1000
    clipping_threshold_per_sample = 5.0
elif type == 'large':
    hidden_dim = 1500
    num_layers = 2
    num_epochs = 55
    alpha = 0.75
    learning_rate = 0.001
    softmax_sample_size = 1000
    clipping_threshold_per_sample = 10.0

num_samples_between_progress_report = 1000
num_words_to_use_in_progress_print = 50

use_sampled_softmax = True
use_sparse = use_sampled_softmax


# Creates model subgraph computing cross-entropy with softmax.
def cross_entropy_with_full_softmax(
    hidden_vector,  # Node providing the output of the recurrent layers
    target_vector,  # Node providing the expected labels (as sparse vectors)
    vocab_dim,      # Vocabulary size
    hidden_dim      # Dimension of the hidden vector
    ):
    bias = C.Parameter(shape = (vocab_dim, 1), init = C.init_bias_default_or_0)
    weights = C.Parameter(shape = (vocab_dim, hidden_dim), init = C.init_default_or_glorot_uniform)

    z = C.reshape( C.times_transpose(weights, hidden_vector) + bias, (1,vocab_dim))
    zT = C.times_transpose(z, target_vector)
    ce = C.reduce_log_sum(z) - zT
    zMax = C.reduce_max(z)
    error_on_samples = C.less(zT, zMax)
    return (z, ce, error_on_samples)

# Creates model subgraph computing cross-entropy with sampled softmax.
def cross_entropy_with_sampled_softmax(
    hidden_vector,           # Node providing the output of the recurrent layers
    target_vector,           # Node providing the expected labels (as sparse vectors)
    vocab_dim,               # Vocabulary size
    hidden_dim,              # Dimension of the hidden vector
    num_samples,             # Number of samples to use for sampled softmax
    sampling_weights,        # Node providing weights to be used for the weighted sampling
    allow_duplicates = False # Boolean flag to control wheather to use sampling with replacemement (allow_duplicates == True) or without replacement.
    ):
    bias = C.Parameter(shape = (vocab_dim, 1), init = C.init_bias_default_or_0)
    weights = C.Parameter(shape = (vocab_dim, hidden_dim), init = C.init_default_or_glorot_uniform)

    sample_selector_sparse = C.random_sample(sampling_weights, num_samples, allow_duplicates) # sparse matrix [num_samples * vocab_size]
    if use_sparse:
        sample_selector = sample_selector_sparse
    else:
        # Note: Sampled softmax with dense data is only supported for debugging purposes.
        # It might easily run into memory issues as the matrix 'I' below might be quite large.
        # In case we wan't to a dense representation for all data we have to convert the sample selector
        I = C.Constant(np.eye(vocab_dim, dtype=np.float32))
        sample_selector = C.times(sample_selector_sparse, I)

    inclusion_probs = C.random_sample_inclusion_frequency(sampling_weights, num_samples, allow_duplicates) # dense row [1 * vocab_size]
    log_prior = C.log(inclusion_probs) # dense row [1 * vocab_dim]

    wS = C.times(sample_selector, weights) # [num_samples * hidden_dim]
    zS = C.times_transpose(wS, hidden_vector) + C.times(sample_selector, bias) - C.times_transpose (sample_selector, log_prior)# [num_samples]

    # Getting the weight vector for the true label. Dimension hidden_dim
    wT = C.times(target_vector, weights) # [1 * hidden_dim]
    zT = C.times_transpose(wT, hidden_vector) + C.times(target_vector, bias) - C.times_transpose(target_vector, log_prior) # [1]


    zSReduced = C.reduce_log_sum(zS)

    # Compute the cross entropy that is used for training.
    # We don't check whether any of the classes in the random samples conincides with the true label, so it might happen that the true class is counted
    # twice in the normalising demnominator of sampled softmax.
    cross_entropy_on_samples = C.log_add_exp(zT, zSReduced) - zT

    # For applying the model we also output a node providing the input for the full softmax
    z = C.times_transpose(weights, hidden_vector) + bias
    z = C.reshape(z, shape = (vocab_dim))

    zSMax = C.reduce_max(zS)
    error_on_samples = C.less(zT, zSMax)
    return (z, cross_entropy_on_samples, error_on_samples)

def create_model(input_sequence, label_sequence, vocab_dim, hidden_dim):
    # Create the rnn that computes the latent representation for the next token.
    rnn_with_latent_output = Sequential([
        C.Embedding(hidden_dim),      
        LayerStack(num_layers, lambda: 
                   Sequential([Stabilizer(), Recurrence(LSTM(hidden_dim), go_backwards=False)]))
        ])

    
    # Apply it to the input sequence. 
    latent_vector = rnn_with_latent_output(input_sequence)

    # Connect the latent output to (sampled/full) softmax.
    if use_sampled_softmax:
        weights = load_sampling_weights(Paths.frequencies)
        smoothed_weights = np.float32( np.power(weights, alpha))
        sampling_weights = C.reshape(C.Constant(smoothed_weights), shape = (1,vocab_dim))
        softmax_input, ce, errs = cross_entropy_with_sampled_softmax(latent_vector, label_sequence, vocab_dim, hidden_dim, softmax_sample_size, sampling_weights)
    else:
        softmax_input, ce, errs = cross_entropy_with_full_softmax(latent_vector, label_sequence, vocab_dim, hidden_dim)

    return softmax_input, ce, errs

# Computes exp(z[index])/np.sum(exp[z]) for a one-dimensional numpy array in an numerically stable way.
def log_softmax(z,    # numpy array
                index # index into the array
            ):
    max_z = np.max(z)
    return z[index] - max_z - log(np.sum(np.exp(z - max_z)))

# Computes the average cross entropy (in nats) for the specified text
def compute_average_cross_entropy(
    model_node,           # node computing the inputs to softmax
    word_sequences,       # List of word sequences
    word_to_id
    ):
    
    id_of_sentence_start =  word_to_id['<s>']

    total_cross_entropy = 0.0
    word_count = 0
    for word_sequence in word_sequences:
        word_ids = [word_to_id[word] for word in word_sequence]

        input = C.one_hot([[word_ids[0]]], len(word_to_id))
        arguments = (input, [True])

        for word_id in word_ids[1:]:
            z = model_node.eval(arguments).flatten()
            # temporary logging as thwere where out of index issues with recent  build !!!!!!!!!!!!!!!!!!!!!
            if len(z) != 10001:
                print("len(z) "+str(len(z))+ "word_id="+ str(word_id))

            log_p = log_softmax(z, word_id)
            total_cross_entropy -= log_p
            input = C.one_hot([[int(word_id)]], len(word_to_id))
            arguments = (input, [False])

        word_count += len(word_ids) - 1
        if word_count >= num_words_to_use_in_progress_print:
            break

    return total_cross_entropy / word_count


# Sample sequences (texts) from the trained model.
def sample(
    model_node,       # Node computing the inputs to softmax.
    id_to_word,       # Dictionary mapping indices to tokens.
    word_to_id,       # Dictionary mapping tokens to indices.
    vocab_dim,        # Size of vocabulary.
    prime_text='',    # Text for priming the model.
    length=100,       # Length of sequence to generate.
    alpha=1.0         # Scaling-exponent used for tweaking the probablities coming from the model. Lower value --> higher diversity.
    ):

    def sample_word_index(p):

        # normalize probabilities then take weighted sample
        p = np.ravel(p)
        p = np.exp(alpha* (p-np.max(p)))
        p = p/np.sum(p)
        word_index = np.random.choice(range(vocab_dim), p=np.ravel(p))
        return word_index

    plen = 1
    prime = -1

    # start sequence with first input
    if prime_text != '':
        words = prime_text.split()
        plen = len(words)
        prime = word_to_id[words[0]]
    else:
        prime = word_to_id['<s>']
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
            idx = word_to_id[words[i+1]]
        else:
            idx = sample_word_index(p)

        output.append(idx)
        x = C.one_hot([[int(idx)]], vocab_dim)

        arguments = (x, [False])
    
    # loop through length of generated text, sampling along the way
    for i in range(length-plen):
        p = model_node.eval(arguments)
        idx = sample_word_index(p)
        output.append(idx)
        x = C.one_hot([[int(idx)]], vocab_dim)

        arguments = (x, [False])

    # return output
    return ' '.join([id_to_word[id] for id in output])

# Creates model inputs
def create_inputs(vocab_dim):
    batch_axis = Axis.default_batch_axis()
    input_seq_axis = Axis('inputAxis')

    input_dynamic_axes = [batch_axis, input_seq_axis]

    input_sequence = input_variable(shape=vocab_dim, dynamic_axes=input_dynamic_axes, is_sparse = use_sparse)
    label_sequence = input_variable(shape=vocab_dim, dynamic_axes=input_dynamic_axes, is_sparse = use_sparse)
    
    return input_sequence, label_sequence

def print_progress(samples_per_second, model, id_to_word, word_to_id, validation_text_file, vocab_dim, total_samples, total_time):
    word_sequences, word_to_id, id_to_word, vocab_dim = load_data_and_vocab(validation_text_file, Paths.token2id)
    print(sample(model, id_to_word, word_to_id, vocab_dim, alpha=1.0, length=10))
    
    average_cross_entropy = compute_average_cross_entropy(model, word_sequences, word_to_id)

    print("time=%.3f ce=%.3f perplexity=%.3f samples=%d samples/second=%.1f" % (total_time, average_cross_entropy, exp(average_cross_entropy), total_samples, samples_per_second))


# Creates and trains an rnn language model.
def train_lm():
    training_text_file = Paths.train
    validation_text_file = Paths.validation
    word_to_id_file_path = Paths.token2id
    sampling_weights_file_path = Paths.frequencies

    # load the data and vocab
    word_sequences, word_to_id, id_to_word, vocab_dim = load_data_and_vocab(training_text_file, word_to_id_file_path)
    

    # Create model nodes for the source and target inputs
    input_sequence, label_sequence = create_inputs(vocab_dim)

    # Create the model. In has three output nodes
    # softmax_input: this provides the latent representation of the next token
    # cross_entropy: this is used training criterion
    # error: this a binary indicator if the model predicts the correct word
    softmax_input, cross_entropy, error = create_model(input_sequence, label_sequence, vocab_dim, hidden_dim)
    
    # Instantiate the trainer object to drive the model training
    lr_per_sample = learning_rate_schedule(learning_rate, UnitType.sample)
    momentum_time_constant = momentum_as_time_constant_schedule(1100)
    gradient_clipping_with_truncation = True
    learner = momentum_sgd(softmax_input.parameters, lr_per_sample, momentum_time_constant, True,
                           gradient_clipping_threshold_per_sample=clipping_threshold_per_sample,
                           gradient_clipping_with_truncation=gradient_clipping_with_truncation)
    trainer = Trainer(softmax_input, cross_entropy, error, learner)

    
    # print out some useful training information
    log_number_of_parameters(softmax_input) ; print()
    
    # Run the training loop
    num_trained_samples = 0
    num_trained_samples_since_last_report = 0

    for epoch_count in range(0, num_epochs):
        num_trained_samples_within_current_epoch = 0
        num_trained_sequences_within_current_epoch = 0

        # Loop over all sequences training data
        for word_sequence in word_sequences:
            t_start = timeit.default_timer()

            # get the data for next sequence (=mini batch)
            features, labels, num_samples = get_data(word_sequence, word_to_id, vocab_dim)
            num_trained_sequences_within_current_epoch += 1
            num_trained_samples_since_last_report += num_samples
            num_trained_samples += num_samples
            num_trained_samples_within_current_epoch += num_samples

            # Specify the mapping of input variables in the model to actual minibatch data to be trained with
            arguments = ({input_sequence : features, label_sequence : labels})
            trainer.train_minibatch(arguments)
            t_end =  timeit.default_timer()
            samples_per_second = num_samples / (t_end - t_start)

            # Print progress report every num_samples_between_progress_report samples
            if num_trained_samples_since_last_report >= num_samples_between_progress_report or num_trained_samples == 0:
                print_progress(samples_per_second, softmax_input, id_to_word, word_to_id, validation_text_file, vocab_dim, num_trained_samples, t_start)
                num_trained_samples_since_last_report = 0

        # store model for current epoch
        model_filename = "models/lm_epoch%d.dnn" % epoch_count
        softmax_input.save_model(model_filename)
        print("Saved model to '%s'" % model_filename)


if __name__=='__main__':


    # train the LM
    train_lm()
