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

    if use_sparse:
        sample_selector = C.random_sample(sampling_weights, num_samples, allow_duplicates) # sparse matrix [num_samples * vocab_size]
    else:
        # in case we wan't to a dense representation for all data we have to convert the sample selector 
        sample_selector_sparse = C.random_sample(sampling_weights, num_samples, allow_duplicates) # sparse matrix [num_samples * vocab_size]
        I = C.Constant(np.eye(vocab_dim, dtype=np.float32))
        sample_selector = C.times(sample_selector_sparse, I)

    inclusion_probs = C.random_sample_inclusion_frequency(sampling_weights, num_samples, allow_duplicates) # dense row [1 * vocab_size]
    log_prior = C.log(inclusion_probs) # dense row [1 * vocab_dim]


    print("hidden_vector: "+str(hidden_vector.shape))
    wS = C.times(sample_selector, weights, name='wS') # [num_samples * hidden_dim]
    print("ws:"+str(wS.shape))
    zS = C.times_transpose(wS, hidden_vector, name='zS1') + C.times(sample_selector, bias, name='zS2') - C.times_transpose (sample_selector, log_prior, name='zS3')# [num_samples]

    # Getting the weight vector for the true label. Dimension hidden_dim
    wT = C.times(target_vector, weights, name='wT') # [1 * hidden_dim]
    zT = C.times_transpose(wT, hidden_vector, name='zT1') + C.times(target_vector, bias, name='zT2') - C.times_transpose(target_vector, log_prior, name='zT3') # [1]

    zSReduced = C.reduce_log_sum(zS)

    # Compute the cross entropy that is used for training.
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
    model_node,          # node computing the inputs to softmax
    word_ids,            # Sequence for which to compute the cross-entropy. Sequence is specified as a list of (word) ids.
    index_of_prime_word,  # Index (of word) to prime the model
    vocab_dim
    ):
    
    priming_input = C.one_hot([[index_of_prime_word]], vocab_dim)
    arguments = (priming_input, [True])

    total_cross_entropy = 0.0
    for word_id in word_ids:
        z = model_node.eval(arguments).flatten()
        log_p = log_softmax(z, word_id)
        total_cross_entropy -= log_p
        x = C.one_hot([[int(word_id)]], vocab_dim)
        arguments = (x, [False])

    return total_cross_entropy / len(word_ids)


# Sample sequences (texts) from the trained model.
def sample(
    model_node,       # Node computing the inputs to softmax.
    ix_to_word,       # Dictionary mapping indices to tokens.
    word_to_ix,       # Dictionary mapping tokens to indices.
    vocab_dim,        # Size of vocabulary.
    prime_text='',    # Text for priming the model.
    use_hardmax=True, # Tells whether hardmax should be used.
    length=100,       # Length of sequence to generate.
    alpha=1.0         # Scaling-exponent used for tweaking the probablities coming from the model. Lower value --> higher diversity.
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

# Creates model inputs
def create_inputs(vocab_dim, asSparse):
    batch_axis = Axis.default_batch_axis()
    input_seq_axis = Axis('inputAxis')

    input_dynamic_axes = [batch_axis, input_seq_axis]

    input_sequence = input_variable(shape=vocab_dim, dynamic_axes=input_dynamic_axes, is_sparse = use_sparse)
    label_sequence = input_variable(shape=vocab_dim, dynamic_axes=input_dynamic_axes, is_sparse = use_sparse)
    
    return input_sequence, label_sequence

def print_progress(samples_per_second, model, ix_to_word, word_to_ix, validation_text_file, vocab_dim, total_samples, total_time):

#    print(sample(model, ix_to_word, word_to_ix, vocab_dim, alpha=0.5))

    id_of_priming_token =  word_to_ix['<unk>']

    # load the test data
    word_ids_test = text_file_to_word_ids(validation_text_file, word_to_ix)
    word_ids_test = word_ids_test[0:num_words_to_use_in_progress_print]
    average_cross_entropy = compute_average_cross_entropy(model, word_ids_test, id_of_priming_token, vocab_dim)
    print("time=%.3f ce=%.3f perplexity=%.3f samples=%d samples/second=%.1f" % (total_time, average_cross_entropy, exp(average_cross_entropy), total_samples, samples_per_second))


# Creates and trains an rnn language model.
def train_lm():
    training_text_file = Paths.train
    validation_text_file = Paths.validation
    word_to_ix_file_path = Paths.token2id
    sampling_weights_file_path = Paths.frequencies

    # load the data and vocab
    data, word_to_ix, ix_to_word, data_size, vocab_dim = load_data_and_vocab(training_text_file, word_to_ix_file_path)
    

    # Model the source and target inputs to the model
    input_sequence, label_sequence = create_inputs(vocab_dim, False)

    # create the node creating the latent vector
    softmax_input, ce, errs = create_model(input_sequence, label_sequence, vocab_dim, hidden_dim)
    
    # Instantiate the trainer object to drive the model training
    lr_per_sample = learning_rate_schedule(learning_rate, UnitType.sample)
    momentum_time_constant = momentum_as_time_constant_schedule(1100)
    gradient_clipping_with_truncation = True
    learner = momentum_sgd(ce.parameters, lr_per_sample, momentum_time_constant, 
                           gradient_clipping_threshold_per_sample=clipping_threshold_per_sample,
                           gradient_clipping_with_truncation=gradient_clipping_with_truncation)
    trainer = Trainer(softmax_input, ce, errs, learner)

    minibatches_per_epoch = int((data_size / minibatch_size))
    total_num_minibatches = num_epochs * minibatches_per_epoch
    
    # print out some useful training information
    log_number_of_parameters(ce) ; print()
    
    epoche_count = 0
    num_trained_samples = 0
    num_trained_samples_within_current_epoche = 0
    num_trained_samples_since_last_report = 0

    for i in range(0, total_num_minibatches):
        t_start = timeit.default_timer()
        if num_trained_samples_within_current_epoche + minibatch_size+1 >= data_size:
            num_trained_samples_within_current_epoche = 0
            epoche_count += 1
            model_filename = "models/lm_epoch%d.dnn" % epoche_count
            softmax_input.save_model(model_filename)
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
        t_end =  timeit.default_timer()
        samples_per_second = minibatch_size / (t_end - t_start)

        if num_trained_samples_since_last_report >= num_samples_between_progress_report or num_trained_samples == 0:
            print_progress(samples_per_second, softmax_input, ix_to_word, word_to_ix, validation_text_file, vocab_dim, num_trained_samples, t_start)
            num_trained_samples_since_last_report = 0

        num_trained_samples += minibatch_size
        num_trained_samples_since_last_report += minibatch_size
        num_trained_samples_within_current_epoche += minibatch_size

# Loads the model from the specified file, samples a sequence of specfied length and writes it to the output file.
def load_and_sample(
        model_filename,         # Relative path of model file
        word_to_ix_file_path,   # Relative path of word index file
        prime_text = '',        # Text to b used for priming the model
        use_hardmax = False,    # Tells wehter hardmax should be used
        length = 100,           # Length of the sample to be generated
        alpha = 1               # Scaling exponent used for tweaking the probablities coming from the model. Used to control diversity of generated sequences.
        ):
    
    # load the model
    model = load_model(model_filename)
    
    # load the vocab
    word_to_ix, ix_to_word = load_word_to_ix(word_to_ix_file_path)
    
    output = sample(model, ix_to_word, word_to_ix, len(word_to_ix), prime_text=prime_text, use_hardmax=use_hardmax, length=length, alpha=alpha)
    
    ff = open('output.txt', 'w', encoding='utf-8')
    ff.write(output)
    ff.close()

if __name__=='__main__':

    # model sizes similar to https://arxiv.org/pdf/1409.2329.pdf and https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py
    type = 'medium'

    if type == 'small':
        hidden_dim = 200
        num_layers = 2
        num_epochs = 150
        minibatch_size = 20
        alpha = 0.75
        learning_rate = 0.003
        softmax_sample_size = 1000
        clipping_threshold_per_sample = 5.0
    elif type == 'medium':
        hidden_dim = 650
        num_layers = 2
        num_epochs = 39
        minibatch_size = 35
        alpha = 0.75
        learning_rate = 0.003
        softmax_sample_size = 1000
        clipping_threshold_per_sample = 5.0
    elif type == 'large':
        hidden_dim = 1500
        num_layers = 2
        num_epochs = 55
        minibatch_size = 35
        alpha = 0.75
        learning_rate = 0.001
        softmax_sample_size = 1000
        clipping_threshold_per_sample = 10.0

    num_samples_between_progress_report = 1000
    num_words_to_use_in_progress_print = 500

    use_sampled_softmax = True
    use_sparse = use_sampled_softmax

    #    set_default_device(cpu()) # this version seem to work fine
    set_default_device(gpu(0)) # this verversion gives nan results

    # work-arround for some bug that seems to affect gradient_accumulation_optimization when dealing wioth sparse
    import _cntk_py
    _cntk_py.disable_gradient_accumulation_optimization()

    # train the LM
    train_lm()
