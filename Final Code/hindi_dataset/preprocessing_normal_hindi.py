from HindiTokenizer import Tokenizer
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, f1_score
import re


regex = re.compile('[,\.!?|#@;:!]')
hindi_stopwords = set()
hw =[]
file1 = open("hindi_stopwords.txt","r", encoding='utf-8')
for line in file1:
    hw = hw + [line.lower().rstrip()]
hindi_stopwords = set(hw)
file1.close()


np.random.seed(500)
header_list = ["text_id","text","task_1","task_2","task_3"]

Corpus = pd.read_csv("train_hindi.tsv",encoding='utf-8', sep="\t",names=header_list)
Corpus['text'] = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', entry) for entry in Corpus['text']]
Corpus['text'] = [regex.sub(' ', entry) for entry in Corpus['text']]
Corpus['text'] = [entry.split() for entry in Corpus['text']]


for index,entry in enumerate(Corpus['text']):
    Final_words = []
    for word in entry:
        if word not in hindi_stopwords:
            t = Tokenizer()
            Final_words.append(t.generate_stem_words(word))
    Corpus.loc[index,'text_final'] = str(Final_words)



Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['task_1'],test_size=0.3)
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
print("SVM F1 Score Task1-> ",f1_score(predictions_SVM, Test_Y, average='weighted')*100)


Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['task_2'],test_size=0.3)
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
print("SVM F1 Score Task2-> ",f1_score(predictions_SVM, Test_Y, average='weighted')*100)

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['task_3'],test_size=0.3)
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
print("SVM F1 Score Task3-> ",f1_score(predictions_SVM, Test_Y, average='weighted')*100)
