# Build Neural Language Model using Sampled Softmax

This example demonstrates how to use sampled softmax for training a token based neural language model.
The model predicts the next word in a text given the previous one where the probablity of the next word is computed using a softmax.
As the number of different words might be very high this final softmax step can turn out to be costly.

Sampled-softmax is a technique to reduce this cost at training time. For details see e.g. (http://www.tensorflow.org/extras/candidate_sampling.pdf)

Note the provided data set has only 10.000 distinct words. This number is till not very high and sampled softmax doesn't show any signficant per improvements here.
The real per gains will show up with larger vocabualries.

## HOWTO

This example uses Penn Treebank Data which is not stored in GitHub but must be downloaded first.
To download the data please run download_data.py once. This will create a directory ./ptb that contains all the data we need 
for running the example.

Run run word-rnn.py to train a model.
The main section of word-rnn defines some parameters to controll the training.

* `use_sampled_softmax` allows to switch between sampled-softmax and full softmax.
* `softmax_sample_size` sets the number of random samples used in sampled-softmax. 

