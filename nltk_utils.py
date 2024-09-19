import nltk
import numpy as np


nltk.download('punkt')

#you did not download this file
nltk.download('punkt_tab')

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


# Function to tokenize the sentence
def tokenize(sentence):
    return nltk.word_tokenize(sentence)


# Function to stem a word
def stem(word):
    return stemmer.stem(word.lower())


# Function to create bag of words
def bag_of_words(tokenize_sentence, all_words):
    # Stemming each word in the tokenized sentence
    tokenize_sentence = [stem(w) for w in tokenize_sentence]

    # Initialize bag of words with zeros
    bag = np.zeros(len(all_words), dtype=np.float32)

    # Mark index as 1 if the word exists in the sentence
    for idx, w in enumerate(all_words):
        if w in tokenize_sentence:
            bag[idx] = 1.0

    return bag


# Example sentence and words
sentence = ["hello", "how", "are", "you"]
words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]

# Get the bag of words
bog = bag_of_words(sentence, words)
print(bog)