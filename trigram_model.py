import sys
from collections import defaultdict
import math
import random
import os
import os.path


def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
  
    sequence = (["START"]* (n-1)) + sequence + ["STOP"]
    ngrams = []
    
    for i in range(len(sequence) - n + 1):
        ngram = tuple(sequence[i:i + n])
        ngrams.append(ngram)
    
    return ngrams

class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)
        total = 0
        for value in self.unigramcounts.values():
            total += value
        self.numberwords = total
        self.numberuniquewords = len(self.unigramcounts)
        


    def count_ngrams(self, corpus):
      
        self.unigramcounts = defaultdict(int) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)
        for sentence in corpus: 
            self.unigramcounts[("START",)] += 1
            self.bigramcounts[("START", "START")] += 1
            
            for unigram in get_ngrams(sentence,1):
                self.unigramcounts[unigram] += 1
            for bigram in get_ngrams(sentence, 2):
                self.bigramcounts[bigram] += 1
            for trigram in get_ngrams(sentence, 3):
                self.trigramcounts[trigram] += 1
                
        ##Your code here

        return

    def raw_trigram_probability(self,trigram):
       
        numtimestrigram = self.trigramcounts.get(trigram, 0)
        bigram = (trigram[0], trigram[1])
        numtimesbigram = self.bigramcounts.get(bigram, 0)
        if numtimesbigram == 0:
            return 1 / self.numberuniquewords
        probs = numtimestrigram/ numtimesbigram
        return probs


        
        

    def raw_bigram_probability(self, bigram):
       
        numtimesbigram = self.bigramcounts.get(bigram, 0)
        unigram = (bigram[0],)
        numtimesunigram = self.unigramcounts.get(unigram, 0)
        if numtimesunigram == 0:
            return 1/self.numberuniquewords
        probs = numtimesbigram / numtimesunigram
        
        
        
        return probs
    
    def raw_unigram_probability(self, unigram):
         
        if unigram not in self.unigramcounts:
            unigram = ("UNK", )
        numtimes = self.unigramcounts.get(unigram, 0)
        if self.numberwords > 0:
            probs = numtimes / self.numberwords 
        else:
            probs = 0
        return probs
        

    def generate_sentence(self,t=20): 
      
        return result            

    def smoothed_trigram_probability(self, trigram):
       
        unigram = (trigram[2], )
        unigramprobs = self.raw_unigram_probability(unigram)
        bigram = (trigram[1], trigram[2])
        bigramprobs = self.raw_bigram_probability(bigram)
        trigram = (trigram[0], trigram[1], trigram[2])
        trigramprobs = self.raw_trigram_probability(trigram)
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        return lambda1*unigramprobs + lambda2*bigramprobs + lambda3*trigramprobs
        
    def sentence_logprob(self, sentence):
        
        probs = 0.0
        trigrams = get_ngrams(sentence, 3)
        for trigram in trigrams:
            triprob = self.smoothed_trigram_probability(trigram)
            

            prob = math.log2(triprob)
            probs += prob
        return probs

    def perplexity(self, corpus):
        
        logprobtotal = 0.0
        wordcount = 0
        
        for sentence in corpus:
            logprobtotal += self.sentence_logprob(sentence)
            wordcount += len(sentence)
        retperplexity = 2 **(-logprobtotal / wordcount)
        return retperplexity
        
        
        
       


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            # .. 
            total += 1
            if pp2 >= pp1:
                correct += 1
                
    
        for f in os.listdir(testdir2):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            # .. 
            total += 1
            if pp1 >= pp2:
                correct += 1
        return float(correct) / total
        

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 
    
   
    

    # Testing perplexity: 
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)


    # Essay scoring experiment: 
    acc = essay_scoring_experiment("ets_toefl_data/train_high.txt", "ets_toefl_data/train_low.txt", "ets_toefl_data/test_high", "ets_toefl_data/test_low")
    print(acc)


