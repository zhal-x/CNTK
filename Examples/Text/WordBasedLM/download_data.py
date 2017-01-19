# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# The program downloads training, validation and test data and creates additional files 
# with token-ids and frequencies fro it.
import urllib.request, os, sys, tarfile, operator

# accumulate word counts in dictionary
def add_to_count(word, word2Count):
    if word in word2Count:
        word2Count[word] += 1
    else:
        word2Count[word] = 1

# for a text file returns a dictionary with the frequency of each word
def count_words_in_file(path):
    f=open(path,'r')
    word2count = {}
    for line in f:
        words = line.split()
        for word in words:
            add_to_count(word, word2count)
    return word2count

# from a dictionary mapping words to counts creates two files: 
# * a vocabulary file containing all words sorted by drecreasing frequency, one word per line
# * a frequency file containg the frequencies of these word, one number per line.
def write_vocab_and_frequencies(word2count, vocab_file_path, freq_file_path, word2count_file_path, word2id_file_path):
    vocab_file = open(vocab_file_path,'w', newline='\r\n')
    freq_file = open(freq_file_path,'w', newline='\r\n')
    word2count_file = open(word2count_file_path,'w', newline='\r\n')
    word2id_file = open(word2id_file_path,'w', newline='\r\n')
    sorted_entries = sorted(word2count.items(), key = operator.itemgetter(1) , reverse = True)
    
    id=int(0)
    for word, freq in sorted_entries:
        vocab_file.write(word+"\n")
        freq_file.write("%i\n" % freq)
        word2count_file.writelines("%s\t%i\n" % (word, freq))
        word2id_file.writelines("%s\t%i\n" % (word, id))
        id +=1

    #close the files
    vocab_file.close()
    freq_file.close()
    word2count_file.close()

class Paths(object):
    data_dir   = './ptb/'

    test_sub       = './simple-examples/data/ptb.test.txt'
    train_sub      = './simple-examples/data/ptb.train.txt'
    validation_sub = './simple-examples/data/ptb.valid.txt'

    test       = os.path.join(data_dir, test_sub)
    train      = os.path.join(data_dir, train_sub)
    validation = os.path.join(data_dir, validation_sub)

    tokens          = os.path.join(data_dir, 'ptb.vocab.txt')
    frequencies     = os.path.join(data_dir, 'ptb.freq.txt')
    token2frequency = os.path.join(data_dir, 'ptb.token2freq.txt')
    token2id        = os.path.join(data_dir, 'ptb.freq2token.txt')


if __name__=='__main__':

    # downloading the data
    url ="http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"
    tmpGz = "tmp.tgz"
    if not os.path.isfile(tmpGz):
        print("downloading " + url + " to " + tmpGz)
        urllib.request.urlretrieve(url, tmpGz)

    # extracting the files we need from the tarfile
    fileReader=tarfile.open(tmpGz, 'r') 
    print("Extracting files into: " + Paths.data_dir)
    fileReader.extract(Paths.test_sub,       path = Paths.data_dir)
    fileReader.extract(Paths.train_sub,      path = Paths.data_dir)
    fileReader.extract(Paths.validation_sub, path = Paths.data_dir)

    fileReader.close()

    #removing the temporary file
    os.remove(tmpGz)

    # from the training file generate a number of helper files
    word2count = count_words_in_file(Paths.train)
    write_vocab_and_frequencies(word2count, Paths.tokens, Paths.frequencies, Paths.token2frequency, Paths.token2id)