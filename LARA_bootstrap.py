__author__ = 'zhangyin'

import numpy as np
import os
import nltk
from nltk.stem.porter import *
from nltk import FreqDist
import math

stemmer = PorterStemmer()
os.chdir("/Users/zhangyin/python projects/IR project")


def parse_to_sentence_UseVocab(content,Vocab,VocabDict):
    sent_word=[]
    sentences = nltk.sent_tokenize(content)
    for sent in sentences:
        words = nltk.word_tokenize(sent)
        temp = [stemmer.stem(w.lower()) for w in words if stemmer.stem(w.lower()) in Vocab]
        temp2 = [VocabDict.get(w) for w in temp]
        if len(temp2)>0:
            sent_word.append(temp2)
    return sent_word

# test code:
# sent = parse_to_sentence_UseVocab(content,Vocab)
# sent

class Review:
    def __init__(self, review_data, Vocab, VocabDict):
        self.Ratings = review_data.get("Ratings")
        self.ReviewID = review_data.get("ReviewID")
        self.Title = review_data.get("Title")
        self.AuthorLocation = review_data.get("AuthorLocation")
        self.Author = review_data.get("Author")
        self.Date = review_data.get("Date")
        self.Content_sentence = parse_to_sentence_UseVocab(review_data.get("Content"),Vocab,VocabDict)
        self.NumOfSents = len(self.Content_sentence)

class Hotel:
    def __init__(self, hotel_data, Vocab, VocabDict):
        self.HotelID = hotel_data.get('HotelInfo').get('HotelID')
        self.Name = hotel_data.get('HotelInfo').get('Name')
        self.Price = hotel_data.get('HotelInfo').get('Price')
        self.Address = hotel_data.get('HotelInfo').get('Address')
        self.ImgURL = hotel_data.get('HotelInfo').get('ImgURL')
        self.HotelURL = hotel_data.get('HotelInfo').get('HotelURL')
        self.Reviews = [Review(review, Vocab,VocabDict) for review in hotel_data.get("Reviews") ]#hotel_data.get("Reviews")
        self.NumOfReviews = len(self.Reviews)

    # def add_reviews(self,hotel_data, Vocab):
    #     for review_data in hotel_data.get("Reviews"):
    #         self.Reviews.append(Review(review_data, Vocab, VocabDict))

def compare_label(label,l):
    return label in l
    # if type(l) is list:
    #     return label in l
    # else:
    #     return label==l

class Corpus:
    def __init__(self, corpus, Vocab, Count, VocabDict):
        self.Vocab = Vocab
        self.VocabDict = VocabDict
        self.VocabTF = Count
        self.V = len(Vocab)
        self.Aspect_Terms = []
        self.Hotels = [ Hotel(hotel, Vocab, VocabDict) for hotel in corpus]
        self.NumOfHotels = len(corpus)
        self.extract_all_sentence()


    def extract_all_sentence(self):
        self.all_sentences = []
        for hotel in self.Hotels:
            for review in hotel.Reviews:
                self.all_sentences=self.all_sentences + review.Content_sentence

    def sent_aspect_match(self,sent,aspects):  ## one sent and all aspect terms, return hit counts for each aspect term
        count = np.zeros(len(aspects))
        i=0
        for a in aspects:
            for w in sent:
                if w in a:
                    count[i]=count[i]+1
            i=i+1
        return count  # a list of length len(aspects)

    def sentence_label(self):   ### produce a label list
        if len(self.all_sentences)>0 and len(self.Aspect_Terms)>0:
            self.labels=[]
            for sent in self.all_sentences:
                count=self.sent_aspect_match(sent,self.Aspect_Terms)
                if max(count)>0:
                    # self.labels.append(np.argmax(count)) # need to be revised
                    s_label = np.where(np.max(count)==count)[0].tolist()
                    # self.labels.append(s) # with tie

                else:
                    self.labels.append(-1)
        else:
            return "Warning: No sentences or Aspect_Terms are recorded in this corpus"

    def To_One_List(self,lists):
        L=[]
        for l in lists:
            L=L+l
        return L

    def calc_chi_sq(self):
        A=len(self.Aspect_Terms)
        # index = np.where(np.array(self.labels)!=-1)[0]  ### remove the sentence that fail to be assigned to any aspect
        index = np.array([ e !=-1 for e in self.labels])
        # labels = np.array(self.labels)[index]
        labels = np.array([v for v in self.labels if v!=-1])
        sentences = np.array(self.all_sentences)[index]

        if A>0:
            self.C1=np.zeros((A,self.V))
            self.C2=np.zeros((A,self.V))
            self.C3=np.zeros((A,self.V))
            self.C4=np.zeros((A,self.V))
            self.C=np.zeros((A,self.V))

            for (k,v) in FreqDist(self.To_One_List(sentences)).items():
                self.C[:,k]=v

            for label in range(A):
                # sent1=sentences[np.where(labels==label)]
                sent1index = np.array([ compare_label(label,l)  for l in labels])
                sent1=sentences[sent1index]
                # sent0=sentences[np.where(labels!=label)]
                sent0=sentences[~sent1index]

                s1=self.To_One_List(sent1)
                s0=self.To_One_List(sent0)
                for (k,v) in FreqDist(s1).items():
                    self.C1[label,k]=v
                for (k,v) in FreqDist(s0).items():
                    self.C2[label,k]=v

                for w in range(self.V):
                    self.C3[label,w]=sum([w not in s for s in sent1])
                    self.C4[label,w]=sum([w not in s for s in sent0])
                    # for s in sent1:
                    #     if w not in s:
                    #         self.C3[label,w]=self.C3[label,w]+1
                    # for s in sent0:
                    #     if w not in s:
                    #         self.C4[label,w]=self.C3[label,w]+1
                print(label)

            self.Chi_sq = self.C*(self.C1*self.C4-self.C2*self.C3)**2/((self.C1+self.C3)*(self.C2+self.C4)*(self.C1+self.C2)*(self.C3+self.C4))
        else:
            print("Warning: No aspects were pre-specified")

    def print_top_k_term_for_aspects(self):
        for cs in self.Chi_sq:
            print(np.argsort(cs)[:5])

# import numpy as np
# (corpus,Vocab,Count,VocabDict)=np.load("./corpus.npy")

all_data = Corpus(corpus, Vocab, Count,VocabDict)

# import pickle
# with open('all_data.pkl', 'wb') as output:
#     pickle.dump(all_data, output, pickle.HIGHEST_PROTOCOL)

A1=[VocabDict.get(stemmer.stem(w.lower())) for w in ["value","price","quality","worth"]]
A2=[VocabDict.get(stemmer.stem(w.lower())) for w in ["room","suite","view","bed"]]
A3=[VocabDict.get(stemmer.stem(w.lower())) for w in ["location","traffic","minute","restaurant"]]
A4=[VocabDict.get(stemmer.stem(w.lower())) for w in ["clean","dirty","maintain","smell"]]
A5=[VocabDict.get(stemmer.stem(w.lower())) for w in ["stuff","check","help","reservation"]]
A6=[VocabDict.get(stemmer.stem(w.lower())) for w in ["service","food","breakfast","buffet"]]
A7=[VocabDict.get(stemmer.stem(w.lower())) for w in ["business","center","computer","internet"]]

all_data.Aspect_Terms=[A1,A2,A3,A4,A5,A6,A7]


I = 10
for i in range(I):
    all_data.sentence_label()
    all_data.labels
    all_data.calc_chi_sq()
    # cs = all_data.Chi_sq[0]
    # len(cs)
    # cs[np.argsort(cs)[:5]]

    # from pprint import pprint
    # pprint([sorted(cs) for cs in all_data.Chi_sq])

    t=0
    for cs in all_data.Chi_sq:
        x = cs[np.argsort(cs)]
        y = np.array([not math.isnan(v) for v in x])
        id = np.argsort(cs)[y]
        print(np.array(Vocab)[id[len(id)-5:]].tolist())
        all_data.Aspect_Terms[t]=list(set(all_data.Aspect_Terms[t]) | set(id[len(id)-5:].tolist()))
        t=t+1

aw = open('aspect_words.txt', 'w')
for a in all_data.Aspect_Terms:
    aw.write(str(np.array(Vocab)[a].tolist())+"\n")
aw.close()

# len(all_data.all_sentences)
# len(all_data.Hotels[0].Reviews[0].Content_sentence+all_data.Hotels[0].Reviews[1].Content_sentence)
# len(all_data.Hotels[0].Reviews[1].Content_sentence+[])
# for k in range(len(corpus)):
#     print(corpus[k].keys()) # each element of data is a dict with 2 keys: "Reviews" and "HotelInfo",
#                           #  which refers a rest with couples of "Reviews"
# hotel = corpus[0]
# review = hotel.get("Reviews")[0]  # each "Reviews" a list, which means 1915 reviews for this restaurant
# review.keys()  # each review in "Reviews" is a dict
#                                 # with 6 keys ['Ratings', 'ReviewID', 'Title', 'Content', 'AuthorLocation', 'Author', 'Date']
# content = review.get('Content') # the content need to be preprocessed.
# hotel.get('HotelInfo').keys()   # each 'RestaurantInfo' is a dict, with
#             #keys ['HotelURL', 'Name', 'Price', 'Address', 'ImgURL', 'HotelID']
# hotel.get('HotelInfo').get('HotelID')
#
#
# # data preprocessing
# # 1. lower case
# # 2. remove punctuations, stopwords, terms occuring <= 5 times
# # 3. stemming with Porter stemmer
#
# #
# # hotel=corpus[0]
# # review=hotel.get("Reviews")[0]
# # content = review.get('Content')




# def save(self):
#     """save class as self.name.txt"""
#     file = open(self.name+'.txt','w')
#     file.write(cPickle.dumps(self.__dict__))
#     file.close()
#
# def load(self):
#     """try load self.name.txt"""
#     file = open(self.name+'.txt','r')
#     dataPickle = file.read()
#     file.close()
#
#     self.__dict__ = cPickle.loads(dataPickle)

# with open('company_data.pkl', 'wb') as output:
#     company1 = Company('banana', 40)
#     pickle.dump(company1, output, pickle.HIGHEST_PROTOCOL)
#
# with open('company_data.pkl', 'rb') as input:
#     company1 = pickle.load(input)
#     print(company1.name)  # -> banana
#     print(company1.value)  # -> 40
#
#     company2 = pickle.load(input)
#     print(company2.name) # -> spam
#     print(company2.value)  # -> 42
#
# import pickle
# import math
# object_pi = math.pi
# file_pi = open('filename_pi.obj', 'wb')
# pickle.dump(object_pi, file_pi)
#
#
# import pickle
# with open('entry.pickle', 'wb') as f:
#     pickle.dump(all_data, f)
#
# with open('entry.pickle', 'rb') as f:
#     entry = pickle.load(f)
