import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, f1_score

np.random.seed(500)

header_list = ["text_id","text","task_1","task_2","task_3"]

Corpus = pd.read_csv("train.tsv",encoding='latin-1', sep="\t",names=header_list)
Corpus['text'].dropna(inplace=True)
Corpus['text'] = [entry.lower() for entry in Corpus['text']]
Corpus['text']= [word_tokenize(entry) for entry in Corpus['text']]

Corpus2 = pd.read_csv("test.tsv",encoding='latin-1', sep="\t",names=header_list)
Corpus2['text'].dropna(inplace=True)
Corpus2['text'] = [entry.lower() for entry in Corpus2['text']]
Corpus2['text']= [word_tokenize(entry) for entry in Corpus2['text']]

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(Corpus['text']):
    Final_words = []
    # main block......
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    Corpus.loc[index,'text_final'] = str(Final_words)


tag_map2 = defaultdict(lambda : wn.NOUN)
tag_map2['J'] = wn.ADJ
tag_map2['V'] = wn.VERB
tag_map2['R'] = wn.ADV
for index,entry in enumerate(Corpus2['text']):
    Final_words = []
    # main block......
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map2[tag[0]])
            Final_words.append(word_Final)
    Corpus2.loc[index,'text_final'] = str(Final_words)


Train_X = Corpus['text_final']
Test_X = Corpus2['text_final']
Train_Y = Corpus['task_1']
Test_Y = Corpus['task_1']

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['text_final'])


Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

# svm mein fit kar Train_X_Tfidf and Train_Y .....
# Train_X_Tfidf hai woh table jiska photo bheja tha
# Train_Y results hai task_1 ke jo fit karega tuh... svm mein

# yeh print functions use karke dekh liyo ki keisa structure hai..
# print(Train_X_Tfidf)
# print(Test_X_Tfidf)

# iske baad svm model se predict kar results Test_X_Tfidf ke....
# predictions and Test_Y ka f1_score(predictions_SVM, Test_Y, average='macro')*100)
# predictions and Test_Y ka f1_score(predictions_SVM, Test_Y, average='weighted')*100)


