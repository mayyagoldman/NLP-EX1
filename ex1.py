import spacy
from datasets import load_dataset

class Corpus():
    unigram_data = {}
    bigram_data  = {}

    def load_data(self):
        nlp = spacy.load("en_core_web_sm")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
        for text in dataset['text']:
            doc = nlp(text)
            for w in doc:
                if w.is_alpha:
                    if w in self.unigram_data:
                        self.unigram_data[w.lemma_] += 1
                    else:
                        self.unigram_data[w.lemma_] = 1
corpus = Corpus()
corpus.load_data()



print(corpus.unigram_data)




"""1. Train maximum-likelihood unigram and bigram language models based on the above training data."""


""" 2. Using the bigram model, continue the following sentence with the most probable word predicted by the model: “ I have a house in ..."""

""" 3. Using the bigram model:
(a) compute the probability of the following two sentences (for each sentence separately).
(b) compute the perplexity of both the following two sentences (treating them as a single test set with 2 sentences).

Brad Pitt was born in Oklahoma
The actor was born in USA
"""

""" 4. Now we use linear interpolation smoothing between the bigram model and unigram model with λbigram = 2/3 and λunigram = 1/3, using the same training data.
 Given this new model, compute the probability and the perplexity of the same sentences such as in the previous question."""