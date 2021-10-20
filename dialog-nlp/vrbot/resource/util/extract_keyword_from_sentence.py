
import sys
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

import os
#import codecs
from textrank4zh import TextRank4Keyword, TextRank4Sentence



def get_abstract(text,num_return=3):
    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, source = 'all_filters')
    print( '摘要：' )
    abstract=[]
    for item in tr4s.get_key_sentences(num=num_return):
        print(item.index, item.weight, item.sentence)
        abstract.append(item.sentence)
    return abstract


def get_keywords(sentence,window=2,lower=True,num_keywords=3):
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=sentence, lower=lower, window=window)
    res=[]
    for item in tr4w.get_keywords(num_keywords, word_min_len=2):
        #print(item.word, item.weight)
        res.append(item.word)
    return res

