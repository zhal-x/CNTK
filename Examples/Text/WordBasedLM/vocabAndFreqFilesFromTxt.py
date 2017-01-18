import operator

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

word2count = count_words_in_file("ptbdata/ptb.train.txt")
write_vocab_and_frequencies(word2count, "ptbdata/ptb.vocab.txt", "ptbdata/ptb.freq.txt", "ptbdata/ptb.word2freq.txt", "ptbdata/ptb.word2id.txt")