import pickle
from nltk.stem.cistem import Cistem
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, f1_score
import re
from IPython.display import display

np.random.seed(500)

#atrate replaces al @ in the text
swear_words = set([])
hate_words = set([])
gender_words = set([])
race_words = set([])
origin_words = set([])
disability_words = set([])
religion_words = set([])
orientation_words = set([])
ethnicity_words = set([])

def parse_files():
    global swear_words,hate_words,gender_words,race_words,origin_words,disability_words,religion_words,orientation_words,ethnicity_words
    hw =[]
    file1 = open("hate.txt","r", encoding='utf-8')
    for line in file1:
        hw = hw + [line.lower().rstrip()]
    hate_words = set(hw)
    file1.close()
    hw =[]
    file1 = open("disability.txt","r", encoding='utf-8')
    for line in file1:
        hw = hw + [line.lower().rstrip()]
    disability_words= set(hw)
    file1.close()
    hw =[]
    file1 = open("ethnicity.txt","r", encoding='utf-8')
    for line in file1:
        hw = hw + [line.lower().rstrip()]
    ethnicity_words = set(hw)
    file1.close()
    hw =[]
    file1 = open("gender.txt","r", encoding='utf-8')
    for line in file1:
        hw = hw + [line.lower().rstrip()]
    gender_words= set(hw)
    file1.close()
    hw =[]
    file1 = open("origin.txt","r", encoding='utf-8')
    for line in file1:
        hw = hw + [line.lower().rstrip()]
    origin_words= set(hw)
    file1.close()
    hw =[]
    file1 = open("race.txt","r", encoding='utf-8')
    for line in file1:
        hw = hw + [line.lower().rstrip()]
    race_words= set(hw)
    file1.close()
    hw =[]
    file1 = open("religion.txt","r", encoding='utf-8')
    for line in file1:
        hw = hw + [line.lower().rstrip()]
    religion_words= set(hw)
    file1.close()
    hw =[]
    file1 = open("sexual.txt","r", encoding='utf-8')
    for line in file1:
        hw = hw + [line.lower().rstrip()]
    orientation_words= set(hw)
    file1.close()
    hw =[]
    file1 = open("swear.txt","r", encoding='utf-8')
    for line in file1:
        hw = hw + [line.lower().rstrip()]
    swear_words= set(hw)
    file1.close()

parse_files()

def country_mile(words,a):
    global swear_words,hate_words,gender_words,race_words,origin_words,disability_words,religion_words,orientation_words,ethnicity_words
    if ((a in swear_words)):
        words = words + ["swear"]

    elif ((a in hate_words)):
        words = words + ["hate"]

    elif ((a in race_words)):
        words = words + ["race"]

    elif ((a in orientation_words)):
        words = words +["orientation"]

    elif ((a in ethnicity_words)):
        words = words + ["ethnicity"]

    elif ((a in gender_words)):
        words = words + ["gender"]

    elif ((a in origin_words)):
        words = words + ["origin"]

    elif ((a in disability_words)):
        words = words + ["disability"]

    elif ((a in religion_words)):
        words = words + ["religion"]

    else :
        z=0
    return words

header_list = ["text_id","text","task_1","task_2"]
regex = re.compile('[,\.!?|#@;:!]')
Corpus = pd.read_csv("train_german.tsv",encoding='latin-1', sep="\t",names=header_list)
Corpus['text'].dropna(inplace=True)
Corpus['text'] = [entry.lower() for entry in Corpus['text']]
Corpus['text'] = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', entry) for entry in Corpus['text']]
Corpus['text'] = [' '.join(re.sub("(@[A-Za-z0-9]+)"," atrate ",entry).split()) for entry in Corpus['text'] ]
Corpus['text'] = [' '.join(re.sub("(#[A-Za-z0-9]+)"," hashtag ",entry).split()) for entry in Corpus['text'] ]
Corpus['text'] = [regex.sub(' ', entry) for entry in Corpus['text']]
Corpus['text'] = [entry.split() for entry in Corpus['text']]
#Corpus['text']= [sent_tokenize(entry, language='german') for entry in Corpus['text']]

for index,entry in enumerate(Corpus['text']):
    Final_words = []
    for word in entry:
        if word not in stopwords.words('german') and word.isalpha():
            if (word == 'atrate'):
                Final_words = Final_words + ["atrate"]
            elif (word == 'hashtag'):
                Final_words = Final_words + ["hashtag"]
            else:
                Final_words = country_mile(Final_words,word)
    Corpus.loc[index,'text_final'] = str(Final_words)
print(Corpus['text_final'])

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['task_1'],test_size=0.3)
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)
Tfidf_vect = TfidfVectorizer(max_features=11)
Tfidf_vect.fit(Corpus['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
pickle.dump(Train_X_Tfidf, open("trainobject", "wb"))
pickle.dump(Test_X_Tfidf, open("testobject", "wb"))
pickle.dump(Train_Y, open("trainobjectsub", "wb"))
pickle.dump(Test_Y, open("testobjectsub", "wb"))


SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
print("SVM F1 Score Task1-> ",f1_score(predictions_SVM, Test_Y, average='weighted')*100)


Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['task_2'],test_size=0.3)
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)
Tfidf_vect = TfidfVectorizer(max_features=11)
Tfidf_vect.fit(Corpus['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
print("SVM F1 Score Task2-> ",f1_score(predictions_SVM, Test_Y, average='weighted')*100)






