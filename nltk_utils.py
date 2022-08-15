import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokeniz(sentence):
  return nltk.word_tokenize(sentence)

def stem(word):
  return stemmer.stem(word.lower())
     
def bag_of_words(tokenized_sentence,words):
  # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1.0

    return bag



# a= "how are you boy" 
# print(a) 
# a=tokenize(a)
# print(a)

# stemming checking
# words = ["organize","organizing","organizers"]
# stemmed_words=[stem(w) for w in words]
# print(stemmed_words)

