# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 10:29:41 2017

@author: Gabrielle Schwartz
"""


from __future__ import print_function
from nltk.stem import *
import numpy as np
import pandas as pd
import os

from html.parser import HTMLParser

html_parser = HTMLParser()

from stop_words import get_stop_words

stop_words = get_stop_words('en')

os.chdir('C:\\Users\schwga02\Google Drive')

shows = pd.read_csv("shows.csv")

episodes = pd.read_csv("episodes-sample.csv")

pd.set_option('display.max_rows', 100)


# codex module -- try and convert to string 

# strip HTML Tags

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()
    
# strip HTML and unescape relevant columns   
    
shows['description'] = shows['description'].astype(str)

shows['cleaned_description'] = shows['description'].apply(strip_tags).\
apply(html_parser.unescape)

shows['title'] = shows['title'].astype(str)

shows['cleaned_title'] = shows['title'].apply(strip_tags).\
apply(html_parser.unescape)

shows['subtitle'] = shows['subtitle'].astype(str)

shows['cleaned_subtitle'] = shows['subtitle'].apply(strip_tags).\
apply(html_parser.unescape)

shows['summary'] = shows['summary'].astype(str)

shows['cleaned_summary'] = shows['summary'].\
apply(strip_tags).apply(html_parser.unescape)


# create new DF with only relevant columns with cleaned columns

shows_cleaned = shows[['id', 'cleaned_description', 'cleaned_title', 'cleaned_summary',
                      'cleaned_subtitle','language', 'feed_url']].copy()
                      

def add_nonequal_columns(*cols):
   result_column = []
   zipped_columns = zip(*cols)
   for vals in zipped_columns:
       text = ''
       vals = set(vals)
       for v in vals:
           text += v + ' '
       result_column.append(text)
   return result_column
   
# Create combined column
   
shows_cleaned['show_text_features'] = add_nonequal_columns(shows_cleaned.cleaned_description, shows_cleaned.cleaned_summary,\
shows_cleaned.cleaned_subtitle)              


# make combined column lower case

shows_cleaned['show_text_features'] = shows_cleaned['show_text_features'].str.lower()

# Final Shows DF ** What is used for combining with episode file

shows_final = shows_cleaned[['id', 'feed_url', 'language', 'show_text_features']].copy()

# write final show file to CSV

shows_final.to_csv( "shows_final.csv")


################### Ignore after this line ################3




# remove duplicate words

def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist
    


########### Vector / matrix creation

# create the stemmer to cut words back to their root
self = PorterStemmer()

#strip the words

def removeStopWords(self,stop_words):
    """ Remove common words which have no search value """
    return [word for word in stop_words if word not in self.stopwords ]


def tokenise(self, string):
    """ break string up into tokens and stem words """
    string = self.clean(string)
    words = string.split(" ")

    return [self.stemmer.stem(word,0,len(word)-1) for word in words]


# map key words to vector dimensions

# create description vector alone

descriptions = episodes['description']


def getVectorKeywordIndex(self, desc):
        """ create the keyword associated to the position of the elements within the document vectors """

        #Mapped documents into a single word string
        vocabularyString = " ".join(desc)

        vocabularyList = self.parser.tokenise(vocabularyString)
        #Remove common words which have no search value
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
                vectorIndex[word]=offset
                offset+=1
        return vectorIndex  #(keyword:position)



#document strings to vectors

def makeVector(self, wordString):
        """ @pre: unique(vectorIndex) """

        #Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        for word in wordList:
                vector[self.vectorKeywordIndex[word]] += 1; #Use simple Term Count Model
        return vector


def cosine(vector1, vector2):
        """ related documents j and q are in the concept space by comparing the vectors :
                cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
        return float(dot(vector1,vector2) / (norm(vector1) * norm(vector2)))
