# Natural Language Processing ----------------------------------------------- #
# --------------------------------------------------------------------------- # NLP Data Cleaning
import numpy as np                                                            # NumPy ~ arrays and matrices
from collections import Counter                                               # dict subclass ~ counts hashable objects
import pandas as pd                                                           # Pandas ~ data frames and series
import nltk                               # Natural Language Tool Kit ------- # nltk libraries help data pre-processing
from nltk.tokenize import word_tokenize                                       # divides text into smaller tokens
from nltk.stem.wordnet import WordNetLemmatizer                               # preferred use of vocabulary for lemma
from nltk.stem import SnowballStemmer                                         # removes word endings for normalization
import string                                                                 # allows finding string.punctuation
from scipy.spatial.distance import pdist, squareform                          # pairwise distance - n-dimensional space
from sklearn.feature_extraction.text import TfidfVectorizer                   # sklearn libraries help model evaluation
from sklearn.metrics.pairwise import cosine_similarity                        # cosine similarity between X and Y
from sklearn.linear_model import LogisticRegression                           # classification via logistic regression
from sklearn.model_selection import train_test_split                          # model ~ test computations
from sklearn.metrics import confusion_matrix                                  # type I and type II errors
#nltk.download('stopwords') # ----------------------------------------------- # uncomment line to download stopwords
#nltk.download('punkt') # --------------------------------------------------- # uncomment line to download punkt
precision = 2                                                                 # desired decimal points in array elements
adj = 10 ** precision                                                         # adjustment of np.array.elements
stops = set(nltk.corpus.stopwords.words('english'))                           # corpus of useless words in nltk
stemmer = SnowballStemmer('english')                                          # corpus of vocabulary for stemming
print(stops)

def our_tokenizer(doc, stops=None, stemmer=None):                             # user defined function our_tokenizer
    doc = word_tokenize(doc.lower())                                          # step 1: covert to lower case
    tokens = [''.join([char for char in tok if char not in string.punctuation]) for tok in doc] # step 2: remove punctua
    tokens = [tok for tok in tokens if tok]
    if stops:
        tokens = [tok for tok in tokens if (tok not in stops)]              # step 3: filter out stopwords from stops
    if stemmer:
        tokens = [stemmer.stem(tok) for tok in tokens]                      # step 4: stemming sandwiches => sandwich
    return tokens

corpus = ['Jim stole my tomato sandwich.',                                     # a list of documents is a corpus
          '"Help!," I sobbed, sandwichlessly.',                                # one text sentence is a document
          '"Drop the sandwiches!," said the sandwich police.']                 # corpus of documents = body of documents

tokenized_docs = [our_tokenizer(doc) for doc in corpus]                        # call our_tokenizer() ~ process corpus

print(corpus[0])                                                               # displays 1st sentence in corpus
print(tokenized_docs[0])                                                       # 1st sentence results of tokenized docs
print(corpus[1])                                                               # displays 2nd sentence in corpus
print(tokenized_docs[1], '  Is i a stopword? ', 'i' in stops)                  # 2nd sentence results, is i in stops?
print(corpus[2])                                                               # displays 3rd sentence in corpus
print(tokenized_docs[2],'  Is drop a stopword? ','drop' in stops)              # 3rd sentence results, is drop in stops?

# ---------------------------------------------------------------------------- # EOF for NLP Data Cleaning
# ---------------------------------------------------------------------------- # NLP count vectorization | TFIDF scores
# ------------------- Count Frequency calculation based on unique tokenized_docs in three sentences(documents) of corpus
import math
vocabulary = tokenized_docs[0] + tokenized_docs[1] + tokenized_docs[2]   # combine all tokenized_docs to vocabulary
vocabulary.sort(reverse=False)                                           # sort vocabulary cleaned list from corpus
print(vocabulary)                                                        # display sorted vocabulary list
print('** Note sandwich duplicated twice ~ 10 unique vocabulary words')  # comment on duplicate words in vocabulary
vocabulary_vector0 = np.array([0,0,1,0,0,1,0,0,1,1])                     # vocabulary vector of unique tokenized_docs[0]
vocabulary_vector1 = np.array([0,1,0,0,0,0,1,1,0,0])                     # vocabulary vector of unqiue tokenized_docs[1]
vocabulary_vector2 = np.array([1,0,0,1,1,2,0,0,0,0])                     # vocabulary vector of unique tokenized_docs[2]
vocabulary_vector  = vocabulary_vector0 + vocabulary_vector1 + vocabulary_vector2 # count vector ~ unique tokenized_docs
cs_cv_0vs1 = cosine_similarity([vocabulary_vector0,vocabulary_vector1])  # assign cosine similaeity array score 0 vs 1
cs_cv_0vs2 = cosine_similarity([vocabulary_vector0,vocabulary_vector2])  # assign cosine similarity array score 0 vs 2
cs_cv_1vs2 = cosine_similarity([vocabulary_vector1,vocabulary_vector2])  # assign cosine similarity array score 1 vs 2
print(tokenized_docs[0],'          ', vocabulary_vector0)              # display first sentence words & vector
print(tokenized_docs[1],'                 ', vocabulary_vector1)       # display second sentence words & vector
print(tokenized_docs[2], vocabulary_vector2)                           # display third sentence words & vector
print('vocabulary word count vector ------------------> ', vocabulary_vector)  # display vocabulary word count vector
print(cs_cv_0vs1, ' <------------cosine similary array vocabulary_vector[0] vs vocabulary_vector[1]')  # cs array scores
print(cs_cv_0vs2, ' <------------cosine similary array vocabulary_vector[0] vs vocabulary_vector[2]')  # cs array scores
print(cs_cv_1vs2, ' <------------cosine similary array vocabulary_vector[1] vs vocabulary_vector[2]')  # cs array scores
# --------------------- Term Frequency calculation = (# times word appears in document) / (total # of words in document)
vocabulary_TFvector0 = np.array([0,0,1/4,0,0,1/4,0,0,1/4,1/4])         # vocabulary TFvector of unique tokenized_docs[0]
vocabulary_TFvector1 = np.array([0,1/3,0,0,0,0,1/3,1/3,0,0])           # vocabulary TFvector of unqiue tokenized_docs[1]
vocabulary_TFvector1 = np.floor(vocabulary_TFvector1 * adj)/adj        # adjust np.array.elements to 2 decimal places
vocabulary_TFvector2 = np.array([1/5,0,0,1/5,1/5,2/5,0,0,0,0])         # vocabulary TFvector of unique tokenized_docs[2]
vocabulary_TFvector = vocabulary_TFvector0 + vocabulary_TFvector1 + vocabulary_TFvector2 # added vocabulary TF vector
vocabulary_TFvector = np.floor(vocabulary_TFvector * adj)/adj        # adjust np.array.elements to 2 decimal places
cs_TF_0vs1 = cosine_similarity([vocabulary_TFvector0,vocabulary_TFvector1])  # cosine similaeity array score 0 vs 1
cs_TF_0vs2 = cosine_similarity([vocabulary_TFvector0,vocabulary_TFvector2])  # cosine similarity array score 0 vs 2
cs_TF_1vs2 = cosine_similarity([vocabulary_TFvector1,vocabulary_TFvector2])  # cosine similarity array score 1 vs 2
print(tokenized_docs[0],'          ', vocabulary_TFvector0)              # display first sentence words & TF vector
print(tokenized_docs[1],'                 ', vocabulary_TFvector1)       # display second sentence words & TF vector
print(tokenized_docs[2], vocabulary_TFvector2)                           # display third sentence words & TF vector
print('vocabulary TF vector --------------------------> ', vocabulary_TFvector)  # display vocabulary word count vector
print(cs_TF_0vs1, ' <------------cosine similarity vocabulary_TFvector[0] vs vocabulary_TFvector[1]')  # cs array scores
print(cs_TF_0vs2, ' <------------cosine similarity vocabulary_TFvector[0] vs vocabulary_TFvector[2]')  # cs array scores
print(cs_TF_1vs2, ' <------------cosine similarity vocabulary_TFvector[1] vs vocabulary_TFvector[2]')  # cs array scores
# --------------------------- Document Frequency calculation = (# of documents containing word) / (total # of documents)
vocabulary_DFvector = np.array([.33,.33,.33,.33,.33,.66,.33,.33,.33,.33]) # vocabulary DF vector ~ unique tokenized_docs
# ------------------ Inverse Document Frequency calculation = log ((total # documents)/(# of documents containing word))
IDFV0 = np.array([0,0,math.log(3),0,0,math.log(1.5),0,0,math.log(3),math.log(3)])        # IDF vector ~ tokenized_docs[0]
IDFV0 = np.floor(IDFV0 * adj)/adj                                        # adjust np.array.elements to 2 decimal places
IDFV1 = np.array([0,math.log(3),0,0,0,0,math.log(3),math.log(3),0,0])                  # IDF vector ~ tokenized_docs[1]
IDFV1 = np.floor(IDFV1 * adj)/adj                                        # adjust np.array.elements to 2 decimal places
IDFV2 = np.array([math.log(3),0,0,math.log(3),math.log(3),math.log(1.5),0,0,0,0])      # IDF vector ~ tokenized_docs[2]
IDFV2 = np.floor(IDFV2 * adj)/adj                                         # adjust np.array.elements to 2 decimal places
vocabulary_IDFvector = IDFV0+IDFV1+IDFV2                                               # vocabulary IDF vector
vocabulary_IDFvector = np.floor(vocabulary_IDFvector * adj)/adj    # adjust np.array.elements to 2 decimal places
cs_IDF_0vs1 = cosine_similarity([IDFV0,IDFV1])  # cosine similaeity array score 0 vs 1
cs_IDF_0vs2 = cosine_similarity([IDFV0,IDFV2])  # cosine similarity array score 0 vs 2
cs_IDF_1vs2 = cosine_similarity([IDFV1,IDFV2])  # cosine similarity array score 1 vs 2
print(vocabulary)                                         # display sorted vocabulary list
print(tokenized_docs[0],'          ', IDFV0)              # display first sentence words & TF vector
print(tokenized_docs[1],'                 ', IDFV1)       # display second sentence words & TF vector
print(tokenized_docs[2], IDFV2)                           # display third sentence words & TF vector
print('vocabulary IDF vector -------------------------> ', vocabulary_IDFvector)  # display vocabulary word count vector
print(cs_IDF_0vs1, ' <--------cosine similarity IDFV[0] vs IDFV[1]')  # cs array scores
print(cs_IDF_0vs2, ' <--------cosine similarity IDFV[0] vs IDFV[2]')  # cs array scores
print(cs_IDF_1vs2, ' <--------cosine similarity IDFV[1] vs IDFV[2]')  # cs array scores
# ----------------------------------------------------------------------- # TFIDF vector = TF vector * IDF vector
vocabulary_TFIDFvector = vocabulary_TFvector * vocabulary_IDFvector       # TFIDF vector of tokenized_docs
vocabulary_TFIDFvector = np.floor(vocabulary_TFIDFvector * adj)/adj       # adjust np.array.elements to 2 decimal places
cs_CV_vs_TF = cosine_similarity([vocabulary_vector,vocabulary_TFvector])            # cosine similarity - count vs TF
cs_CV_vs_DF = cosine_similarity([vocabulary_vector,vocabulary_DFvector])            # cosine similarity ~ count vs DF
cs_CV_vs_IDF = cosine_similarity([vocabulary_vector,vocabulary_IDFvector])          # cosine similarity ~ count vs IDF
cs_CV_vs_TFIDF = cosine_similarity([vocabulary_vector,vocabulary_TFIDFvector])      # cosine similarity ~ count vs TFIDF
# -------------------------------------------------------------------- # display tokenized_docs and vectors
print(vocabulary)
print(tokenized_docs[0],'          ', vocabulary_vector0)              # display first sentence words & vector
print(tokenized_docs[1],'                 ', vocabulary_vector1)       # display second sentence words & vector
print(tokenized_docs[2], vocabulary_vector2)                           # display third sentence words & vector
print('vocabulary word count vector ------------------> ', vocabulary_vector)  # display vocabulary word count vector
print('vocabulary TF vector --------------------------> ', vocabulary_TFvector) # display vocabulary TF vector array
print('vocabulary DF vector --------------------------> ', vocabulary_DFvector) # display vocabulary DF vector array
print('vocabulary IDF vector -------------------------> ', vocabulary_IDFvector) # display vocabulary IDF vector array
print('vocabulary TFIDF vector -----------------------> ', vocabulary_TFIDFvector) # display vocabulary TFIDF vector
print(cs_CV_vs_TF, ' <--------cosine similarity ~ vocabulary count vs vocabulary TF')        # cs array scores
print(cs_CV_vs_DF, ' <--------cosine similarity ~ vocabulary count vs vocabulary DF')        # cs array scores
print(cs_CV_vs_IDF, ' <--------cosine similarity ~ vocabulary count vs vocabulary IDF')      # cs array scores
print(cs_CV_vs_TFIDF, ' <--------cosine similarity ~ vocabulary count vs vocabulary TFIDF')  # cs array scores
# -------------------------------------------------------------------- # EOF for Count Vectorization | TFIDF


