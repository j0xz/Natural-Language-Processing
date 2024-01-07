# Natural Language Processing ----------------------------------------------- #
# --------------------------------------------------------------------------- # NLP SMSCollections
# Natural Language Processing ----------------------------------------------- #
# --------------------------------------------------------------------------- # NLP SMSCollections with tweaking
import numpy as np                                                            # NumPy ~ arrays and matrices
import pandas as pd                                                           # Pandas ~ data frames and series
import nltk                               # Natural Language Tool Kit ------- # nltk libraries help data pre-processing
import string                                                                 # allows finding string.punctuation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  # sklearn libraries help model evaluation
from sklearn.linear_model import LogisticRegression                           # classification via logistic regression
from sklearn.ensemble import RandomForestClassifier                          # classification via RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier                   # classification via GradientBootingClassifier
from sklearn.model_selection import train_test_split                          # model ~ test computations
from sklearn.metrics import confusion_matrix                                  # type I and type II errors
#nltk.download('stopwords') # ----------------------------------------------- # uncomment line to download stopwords
#nltk.download('punkt') # --------------------------------------------------- # uncomment line to download punkt
desired_width = 400
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)
stopwords_set = set(nltk.corpus.stopwords.words('english'))                   # corpus of useless words in nltk
punctuation_set = set(string.punctuation)                                     # corpus of punctuation marks in string
print('# stopwords = ', len(stopwords_set), '     # punctuations = ', len(punctuation_set))      # display stops & punct
print('          ')
# --------------------------------------------------------------------------- # load SMS Collections into a dataframe
data_file = 'SMSCollection'                                                   # assigns file name variable data_file
df = pd.read_table(data_file, header=None)                                    # reads SMSCollections file as a table
df.columns = ('spam', 'message')                                              # assigns column names to df
# ------------------------------------------ # Clean message column by filtering it by stopwords_set and punctuation_set
#                                            # ' '.join adds the cleaned message msg_cleaned as a str
df['msg_cleaned'] = df.message.apply(lambda x:  ' '.join([word for word in x.split() if word not in stopwords_set and \
                                                          word not in punctuation_set]))
df['msg_cleaned'] = df.msg_cleaned.str.lower()                                # converts all characters to lower case
print(df.head(4))
print('                   ')
# ------------------------------------------ # Use CountVectorizer to create unique word vector array X # ------------ #
count_vector = CountVectorizer()                                              # assign variable to CountVectorizer()
X = count_vector.fit_transform(df.msg_cleaned)                                # pass df.msg_cleaned for count vectorizer
print('count vector  X  matrix type = ', type(X), 'X matrix shape = ', X.shape)     # display type and shape of matrix X
y = df.spam                                                            # df.spam column is what to predict (spam or ham)
X_train, X_test, y_train, y_test = train_test_split(X,y)                      # split X,y into train/test random subsets
# ------------------------------------------ # Use TfidfVectorizer to create unique word vector arrays X1,X2 # ------- #
TFIDF_vector = TfidfVectorizer()                                             # assign var to TfidfVectorizer()
TFIDF_vector1 = TfidfVectorizer(ngram_range=(1,3))                           # assign var to TFIDF w/ bigrams & trigrams
X1 = TFIDF_vector.fit_transform(df.msg_cleaned)                              # pass df.msg_cleaned for TFIDF vectorizer
X2 = TFIDF_vector1.fit_transform(df.msg_cleaned)                   # pass df.msg_cleaned for TFIDF w/ bigrams & trigrams
print('TFIDF vector  X1 matrix type = ', type(X1), 'X1 matrix shape = ', X1.shape) # display type and shape of matrix X1
print('TFIDF* vector X2 matrix type = ', type(X2), 'X2 matrix shape = ', X2.shape) # display type and shape of matrix X2
y1 = df.spam                                                           # df.spam column is what to predict (spam or ham)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1,y1)                 # split X,y into train/test random subsets
X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y1)
print('             ')
# ----------------------------------------------------------------------- # run logistic regression and confusion matrix
logreg = LogisticRegression()
logreg1 = LogisticRegression()
logreg2 = LogisticRegression()
logreg.fit(X_train,y_train)
logreg1.fit(X1_train,y1_train)
logreg2.fit(X2_train,y2_train)
y_pred = logreg.predict(X_test)
y1_pred = logreg1.predict(X1_test)
y2_pred = logreg2.predict(X2_test)
logregXR2 = logreg.score(X_test,y_test)
logregX1R2 = logreg1.score(X1_test,y1_test)
logregX2R2 = logreg2.score(X2_test,y2_test)
print('Logistic Regression X R2 ------------> ', logregXR2)
print('Logistic Regression X1 R2 ------------> ', logregX1R2)
print('Logistic Regression X2 R2 ------------> ', logregX2R2)
type_err_matrix = confusion_matrix(y_test,y_pred)
type_err_matrix1 = confusion_matrix(y1_test,y1_pred)
type_err_matrix2 = confusion_matrix(y2_test,y2_pred)
print(type_err_matrix, ' <--------------- Confusion Matrix [y_test, y_pred]')
print(type_err_matrix1, ' <--------------- Confusion Matrix [y1_test, y1_pred]')
print(type_err_matrix2, ' <--------------- Confusion Matrix [y2_test, y2_pred]')
print('          ')                                                                # line separator
# ----------------------------------------------------------------- # run random forest classifier with confusion matrix
rf = RandomForestClassifier()
rf1 = RandomForestClassifier()
rf2 = RandomForestClassifier()
rf.fit(X_train,y_train)
rf1.fit(X1_train,y1_train)
rf2.fit(X2_train,y2_train)
y3_pred = rf.predict(X_test)
y4_pred = rf1.predict(X1_test)
y5_pred = rf2.predict(X2_test)
print('RandomForest Classifier X R2 ----------> ', rf.score(X_test,y_test))        # calculate/display RandomTree X R2
print('RandomForest Classifier X1 R2 ----------> ', rf1.score(X1_test,y1_test))    # calculate/display  RandomTree X1 R2
print('RandomForest Classifier X2 R2 ----------> ', rf2.score(X2_test,y2_test))    # calculate/display RandomTree X2 R2
type_err_matrix3 = confusion_matrix(y_test,y3_pred)                              # calculate Type I & II errors - X,y
type_err_matrix4 = confusion_matrix(y1_test,y4_pred)                             # calculate Type I & II errors - X1,y1
type_err_matrix5 = confusion_matrix(y2_test,y5_pred)                             # calculate Type I & II errors - X2,y2
print(type_err_matrix3, ' <--------------- Confusion Matrix [y_test,y3_pred]')    # display Type I & II errors - X,y
print(type_err_matrix4, ' <--------------- Confusion Matrix [y1_test,y4_pred]')   # display Type I & II errors - X1,y1
print(type_err_matrix5, ' <--------------- Confusion Matrix [y2_test,y5_pred]')   # display Type I & II errors - X2,y2
print('           ')                                                             # line separator
# ---------------------------------------------------------------- # run gradient boost classifier with confusion matrix
gb = GradientBoostingClassifier()
gb1 = GradientBoostingClassifier()
gb2 = GradientBoostingClassifier()
gb.fit(X_train,y_train)
gb1.fit(X1_train,y1_train)
gb2.fit(X2_train,y2_train)
y6_pred = gb.predict(X_test)
y7_pred = gb1.predict(X1_test)
y8_pred = gb2.predict(X2_test)
print('Gradient Boosting Classifier X R2 ----------> ', gb.score(X_test,y_test))
print('Gradient Boosting Classifier X1 R2 ----------> ', gb1.score(X1_test,y1_test))
print('Gradient Boosting Classifier X2 R2 ----------> ', gb2.score(X2_test,y2_test))
type_err_matrix6 = confusion_matrix(y_test,y6_pred)
type_err_matrix7 = confusion_matrix(y1_test,y7_pred)
type_err_matrix8 = confusion_matrix(y2_test,y8_pred)
print(type_err_matrix6, ' <--------------- Confusion Matrix [y_test,y6_pred]')
print(type_err_matrix7, ' <--------------- Confusion Matrix [y1_test,y7_pred]')
print(type_err_matrix8, ' <--------------- Confusion Matrix [y2_test,y8_pred]')
# --------------------------------------------------------------------------------# EOF SMSCollection model tweaking
