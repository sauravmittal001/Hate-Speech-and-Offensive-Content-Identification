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
swear_words = set(["ass","arse","asshole","bastard","bitch","bollocks","bugger","bullshit","crap","cunt","damn","effing","frigger","fuck","fucker","goddamn","godsdamn",
"hell","holyshit","horseshit","motherfucker","nigga","piss","prick","shit","shitass","slut","whore","twat"])
hate_words = set([])
gender_words = set([])
race_words = set([])
origin_words = set([])
disability_words = set([])
religion_words = set([])
orientation_words = set([])


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
    x = split_hashtag(text)
    if len(x)>0:
        return x[0]
    else:
        return []

def country_mile(words,a,b,temp):
    if ((a in swear_words) or (b in swear_words)):
        words = words = ["swear"]
    elif ((a in hate_words) or (b in hate_words)):
        words = words = ["hate"]
    elif ((a in gender_words) or (b in gender_words)):
        words = words = ["gender"]
    elif ((a in race_words) or (b in race_words)):
        words = words = ["race"]
    elif ((a in origin_words) or (b in origin_words)):
        words = words = ["origin"]
    elif ((a in disability_words) or (b in disability_words)):
        words = words = ["disability"]
    elif ((a in religion_words) or (b in religion_words)):
        words = words = ["religion"]
    elif ((a in orientation_words) or (b in orientation_words)):
        words = words = ["orientation"]
    else :
        if (temp == 1):
            words = words = ["proper"]
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
Train_Y = Corpus['task_1']
Test_Y = Corpus2['task_1']

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

Tfidf_vect = TfidfVectorizer(max_features=10)
Tfidf_vect.fit(Corpus['text_final'])

Tfidf_vect2 = TfidfVectorizer(max_features=10)
Tfidf_vect2.fit(Corpus2['text_final'])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect2.transform(Test_X)






