import numpy as np
import spacy
from datasets import load_dataset

PROB = 0
WORD = 1
UNIGRAM = 1
BIGRAM = 2


class Corpus:
    unigram_data = {}  # {w:count}
    bigram_data = {}  # {w_prev: {w:count}}
    unigram_data_size = 0
    bigram_data_size = {}

    def add_to_unigram(self, w):
        self.unigram_data_size += 1
        if w in self.unigram_data:
            self.unigram_data[w] += 1
        else:
            self.unigram_data[w] = 1
        self.unigram_data_size += 1

    def add_to_bigram(self, w, w_prev):
        if w_prev not in self.bigram_data:
            self.bigram_data[w_prev] = {}
            self.bigram_data_size[w_prev] = 0
        self.bigram_data_size[w_prev] += 1
        w_prev_dict = self.bigram_data[w_prev]
        if w in w_prev_dict:
            w_prev_dict[w] += 1
        else:
            w_prev_dict[w] = 1

    def load_data(self):
        nlp = spacy.load("en_core_web_sm")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
        for text in dataset['text']:
            doc = nlp(text)
            w_prev = 'START'
            for w in doc:
                if w.is_alpha:
                    self.add_to_unigram(w.lemma_)
                    self.add_to_bigram(w.lemma_, w_prev)
                    w_prev = w.lemma_  # todo: check first word in extreme cases is indeed START


class NGramModel:

    def __init__(self, corpus, n):
        self.n = n
        if n == 1:
            self.corpus = corpus.unigram_data  # {w: count / N}
            self.max_prob = 0
            self.max_prob_word = None
            self.total_count = corpus.unigram_data_size
        elif n == 2:
            self.corpus = corpus.bigram_data
            self.max_probs = {}
            self.total_counts = corpus.bigram_data_size

    def train_unigram(self):
        for w_i in self.corpus.keys():
            count = self.corpus[w_i]
            prob_w_i = np.log(count / self.total_count)
            self.corpus[w_i] = prob_w_i
            if prob_w_i > self.max_prob:
                self.max_prob = prob_w_i
                self.max_prob_word = w_i

    def train_bigram(self):
        for (w_i, proceeding_w_j) in self.corpus:
            self.max_probs[w_i] = [0, None]
            for (w_j, count) in proceeding_w_j:
                prob_w_j = np.log(count / self.total_counts[w_i])
                proceeding_w_j[w_j] = prob_w_j
                if prob_w_j > self.max_probs[w_i][0]:
                    self.max_probs[w_i][PROB] = prob_w_j
                    self.max_probs[w_i][WORD] = w_j

    def train(self):
        if self.n == 1:
            self.train_unigram()
        elif self.n == 2:
            self.train_bigram()

    def _predict_unigram(self, sentence):
        return self.max_prob_word

    def _predict_bigram(self, sentence):
        last_word = sentence[-1]
        if last_word not in self.corpus:
            return "STOP"
        return self.max_probs[last_word][WORD]

    def predict(self, sentence):
        if self.n == 1:
            return self._predict_unigram(sentence)
        elif self.n == 2:
            return self._predict_bigram(sentence)

    def _probability_unigram(self, sentence):
        prob = 1
        for w in sentence:
            if w not in self.corpus:
                return 0
            prob *= self.corpus[w]
        return prob

    def _probability_bigram(self, sentence):
        sentence = "START " + sentence
        prob = 1
        for i in range(1, len(sentence)):
            w_prev = sentence[i - 1]
            w_i = sentence[i]
            if w_prev not in self.corpus:
                return 0
            if w_i not in self.corpus[w_prev]:
                return 0
            prob *= self.corpus[w_prev][w_i]
        return prob

    def probability(self, sentence):
        if self.n == 1:
            return self._probability_unigram(sentence)
        elif self.n == 2:
            return self._probability_bigram(sentence)

    def perplexity(self):
        pass


class LinearInterpolation:
    def __init__(self, unigram, bigram):
        self.unigram_model = unigram
        self.unigram_model = bigram

    def probability(self, sentence):
        pass

    def perplexity(self):
        pass


"""1. Train maximum-likelihood unigram and bigram language models based on the above training data."""

corpus = Corpus()
corpus.load_data()
# print(corpus.unigram_data)
unigram_model = NGramModel(corpus, UNIGRAM)
bigram_model = NGramModel(corpus, BIGRAM)
unigram_model.train_unigram()
bigram_model.train_bigram()

"""2. Using the bigram model, continue the following sentence with the most probable word predicted by the model: “ I 
have a house in ... """
sentence = 'I have a house in'
predicted_word = bigram_model.predict(sentence)
print(predicted_word)

""" 3. Using the bigram model:
(a) compute the probability of the following two sentences (for each sentence separately).
(b) compute the perplexity of both the following two sentences (treating them as a single test set with 2 sentences).

Brad Pitt was born in Oklahoma
The actor was born in USA
"""
sentence1 = 'Brad Pitt was born in Oklahoma'
sentence2 = 'The actor was born in USA'

sentence1_prob = bigram_model.probability(sentence1)
sentence2_prob = bigram_model.probability(sentence2)
print(f'sentence 1 probability:{sentence1_prob}')
print(f'sentence 2 probability:{sentence2_prob}')

"""4. Now we use linear interpolation smoothing between the bigram model and unigram model with λbigram = 2/3 and 
λunigram = 1/3, using the same training data. Given this new model, compute the probability and the perplexity of the 
same sentences such as in the previous question. """
