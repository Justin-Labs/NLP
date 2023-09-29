import nltk # importing nltk by using "import command than the library name"

# Word Corpora library from NLTK
from nltk.corpus import gutenberg

# Tokenization Library from NLTK
from nltk.tokenize import sent_tokenize, word_tokenize ,regexp_tokenize



# Stemming Library
from nltk.stem import PorterStemmer           
from nltk.stem.lancaster import LancasterStemmer


# Lemmatization Library
from nltk.stem import WordNetLemmatizer

#stopword removal library
from nltk.corpus import stopwords

#Most frequent word removal
from nltk.probability import FreqDist

# Spell Correction
from nltk.metrics import edit_distance

# Part of speech Tagging
from nltk import pos_tag

#N-Gram library
from nltk import ngrams

#NER Library from NLTK
from nltk import ne_chunk

#chunking 
from nltk.chunk.regexp import *