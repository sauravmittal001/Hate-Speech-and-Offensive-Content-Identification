import pickle
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet
from nltk.corpus import words, brown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, f1_score
import re
lemmatizer = WordNetLemmatizer()

np.random.seed(500)

hashtag_dict={}
def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('NNP'):
        return "Proper"
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

#proper nouns
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
word_dictionary = list(set(words.words())) + list(swear_words)
for alphabet in "bcdefghjklmnopqrstuvwxyz":
	word_dictionary.remove(alphabet)

def split_hashtag(hashtag):
    all_possibilities = []
    split_posibility = [hashtag[:i] in word_dictionary for i in reversed(range(len(hashtag)+1))]
    possible_split_positions = [i for i, x in enumerate(split_posibility) if x == True]
    for split_pos in possible_split_positions:
        split_words = []
        word_1, word_2 = hashtag[:len(hashtag)-split_pos], hashtag[len(hashtag)-split_pos:]
        if word_2 in word_dictionary:
            split_words.append(word_1)
            split_words.append(word_2)
            all_possibilities.append(split_words)
            another_round = split_hashtag(word_2)
            if len(another_round) > 0:
                x = [[a1] + a2 for a1, a2, in zip([word_1]*len(another_round), another_round)]
                all_possibilities = all_possibilities + x
        else:
            another_round = split_hashtag(word_2)
            if len(another_round) > 0:
                x = [[a1] + a2 for a1, a2, in zip([word_1]*len(another_round), another_round)]
                all_possibilities = all_possibilities + x
    return all_possibilities

def hashtag(text):
    global hashtag_dict
    if (text in hashtag_dict):
        return hashtag_dict.get(text)
    else:
        x = split_hashtag(text)
        if len(x)>0:
            hashtag_dict[text] = x[0]
            return x[0]
        else:
            hashtag_dict[text] = []
            return []

def country_mile(words,a,b,temp):
    global swear_words,hate_words,gender_words,race_words,origin_words,disability_words,religion_words,orientation_words,ethnicity_words
    if ((a in swear_words) or (b in swear_words)):
        words = words + ["swear"]
    elif ((a in hate_words) or (b in hate_words)):
        words = words + ["hate"]
    elif ((a in race_words) or (b in race_words)):
        words = words + ["race"]
    elif ((a in orientation_words) or (b in orientation_words)):
        words = words +["orientation"]
    elif ((a in ethnicity_words) or (b in ethnicity_words)):
        words = words + ["ethnicity"]
    elif ((a in gender_words) or (b in gender_words)):
        words = words + ["gender"]
    elif ((a in origin_words) or (b in origin_words)):
        words = words + ["origin"]
    elif ((a in disability_words) or (b in disability_words)):
        words = words + ["disability"]
    elif ((a in religion_words) or (b in religion_words)):
        words = words + ["religion"]
    else :
        if (temp == 1):
            words = words + ["proper"]
    return words

header_list = ["text_id","text","task_1","task_2","task_3"]

Corpus = pd.read_csv("train_english.tsv",encoding='latin-1', sep="\t",names=header_list)
Corpus['text'].dropna(inplace=True)
Corpus['text'] = [entry.lower() for entry in Corpus['text']]
Corpus['text'] = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', entry) for entry in Corpus['text']]
Corpus['text'] = [entry.replace(".", " ") for entry in Corpus['text']]
Corpus['text'] = [' '.join(re.sub("(@[A-Za-z0-9]+)"," atrate ",entry).split()) for entry in Corpus['text'] ]
Corpus['text']= [word_tokenize(entry) for entry in Corpus['text']]
main_list=[]
for entry in Corpus['text']:
    add_words =[]
    remove_elements=[]
    l = entry
    for i in range(len(l)):
        if entry[i] == '#':
            remove_elements = remove_elements + [i+1]
            add_words = add_words + hashtag(l[i+1])
    for i in range(len(remove_elements)):
        l.pop(remove_elements[i] -i)
    l = l + add_words
    main_list = main_list + [l]
Corpus['text'] = main_list

for index,entry in enumerate(Corpus['text']):
    Final_words = []
    pos_tagged = nltk.pos_tag(entry)
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    for word, tag in wordnet_tagged:
        if word not in stopwords.words('english') and word.isalpha():
            if (word == "atrate"):
                Final_words = Final_words + ["atrate"]
            else:
                if tag is None:
                    Final_words = country_mile(Final_words,word,word,0)
                elif (tag == "Proper"):
                    worda = lemmatizer.lemmatize(word,wordnet.NOUN)
                    Final_words = country_mile(Final_words,word,worda,1)
                else:
                    worda = lemmatizer.lemmatize(word,tag)
                    Final_words = country_mile(Final_words,word,worda,0)
    Corpus.loc[index,'text_final'] = str(Final_words)


Corpus2 = pd.read_csv("test_english.tsv",encoding='latin-1', sep="\t",names=header_list)
Corpus2['text'].dropna(inplace=True)
Corpus2['text'] = [entry.lower() for entry in Corpus2['text']]
Corpus2['text'] = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', entry) for entry in Corpus2['text']]
Corpus2['text'] = [entry.replace(".", " ") for entry in Corpus2['text']]
Corpus2['text'] = [' '.join(re.sub("(@[A-Za-z0-9]+)"," atrate ",entry).split()) for entry in Corpus2['text'] ]
Corpus2['text']= [word_tokenize(entry) for entry in Corpus2['text']]
main_list=[]
for entry in Corpus2['text']:
    add_words =[]
    remove_elements=[]
    l = entry
    for i in range(len(l)):
        if entry[i] == '#':
            remove_elements = remove_elements + [i+1]
            add_words = add_words + hashtag(l[i+1])
    for i in range(len(remove_elements)):
        l.pop(remove_elements[i] -i)
    l = l + add_words
    main_list = main_list + [l]
Corpus2['text'] = main_list

for index,entry in enumerate(Corpus2['text']):
    Final_words = []
    pos_tagged = nltk.pos_tag(entry)
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    for word, tag in wordnet_tagged:
        if word not in stopwords.words('english') and word.isalpha():
            if word is "atrate":
                Final_words = Final_words + ["atrate"]
            else:
                if tag is None:
                    Final_words = country_mile(Final_words,word,word,0)
                elif tag is "Proper":
                    worda = lemmatizer.lemmatize(word,wordnet.NOUN)
                    Final_words = country_mile(Final_words,word,worda,1)
                else:
                    worda = lemmatizer.lemmatize(word,tag)
                    Final_words = country_mile(Final_words,word,worda,0)
    Corpus2.loc[index,'text_final'] = str(Final_words)


Train_X = Corpus['text_final']
Test_X = Corpus2['text_final']
Train_Y1 = Corpus['task_1']
Test_Y1 = Corpus2['task_1']
Train_Y2 = Corpus['task_2']
Test_Y2 = Corpus2['task_2']
Train_Y3 = Corpus['task_3']
Test_Y3 = Corpus2['task_3']

Encoder = LabelEncoder()
Train_Y1 = Encoder.fit_transform(Train_Y1)
Test_Y1 = Encoder.fit_transform(Test_Y1)
Train_Y2 = Encoder.fit_transform(Train_Y2)
Test_Y2 = Encoder.fit_transform(Test_Y2)
Train_Y3 = Encoder.fit_transform(Train_Y3)
Test_Y3 = Encoder.fit_transform(Test_Y3)

Tfidf_vect = TfidfVectorizer(max_features=11)
Tfidf_vect.fit(Corpus['text_final'])

Tfidf_vect2 = TfidfVectorizer(max_features=11)
Tfidf_vect2.fit(Corpus2['text_final'])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect2.transform(Test_X)

pickle.dump(Train_X_Tfidf, open("trainobject", "wb"))
pickle.dump(Test_X_Tfidf, open("testobject", "wb"))
#me = pickle.load(open("myobject", "rb"))

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y1)
predictions_SVM = SVM.predict(Test_X_Tfidf)
print("SVM F1 Score Task1-> ",f1_score(predictions_SVM, Test_Y1, average='weighted')*100)
print("SVM F1 Score Task1 macro-> ",f1_score(predictions_SVM, Test_Y1, average='macro')*100)

SVM2 = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM2.fit(Train_X_Tfidf,Train_Y2)
predictions_SVM2 = SVM2.predict(Test_X_Tfidf)
print("SVM F1 Score Task2-> ",f1_score(predictions_SVM2, Test_Y2, average='weighted')*100)
print("SVM F1 Score Task2 macro-> ",f1_score(predictions_SVM2, Test_Y2, average='macro')*100)

SVM3 = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM3.fit(Train_X_Tfidf,Train_Y3)
predictions_SVM3 = SVM3.predict(Test_X_Tfidf)
print("SVM F1 Score Task3-> ",f1_score(predictions_SVM3, Test_Y3, average='weighted')*100)
print("SVM F1 Score Task3 macro-> ",f1_score(predictions_SVM3, Test_Y3, average='macro')*100)



