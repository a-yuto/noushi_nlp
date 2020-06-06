#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from gensim import corpora
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import gensim
import MeCab
import pytest

def lda(documents: list,test: list) -> 'LdaModel':
    stop_words = set('for a of the and to in'.split())
    texts = [[word for word in document.lower().split() if word not in stop_words] for document in documents]
    frequency = defaultdict(int)
    for text in texts:
         for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] > 1] for text in texts]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=5, id2word=dictionary)
    test_texts = [[word for word in document.lower().split()] for document in test_documents]
    print(test_texts)
    test_corpus = [dictionary.doc2bow(text) for text in test_texts]
    for topics_per_document in lda[test_corpus]:
        pprint(topics_per_document)
    return lda

def mecabstr_list(text: str) -> list:
    ans_list = []
    tmp = ""
    for ch in text:
        if ch == ",":
            ans_list.append(tmp)
            tmp = ""
        else:
            tmp += ch
    ans_list.append(tmp)
    return ans_list

def wakati(text: str) -> str:
    return MeCab.Tagger("-Owakati").parse(text)

## pp = postpositional particle(助詞)
def wakati_without_pp(text: str) -> str:
    tagger = MeCab.Tagger("-Ochasen")
    tagger.parse("")
    node = tagger.parseToNode(text)
    ans_text = ""
    while node:
        list_node = mecabstr_list(node.feature)
        if list_node[0] == "助詞" or list_node[0] == 'BOS/EOS':
            node = node.next
            continue        
        ans_text += list_node[6] + " "
        node = node.next
    return ans_text + "\n"


def tfidf(curpus: list) -> (list,np.ndarray):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(curpus).toarray()
    columns =  vectorizer.get_feature_names()
    for row in X:
        plt.bar(columns, row)
        plt.show()
    return columns, X



##
##            テスト
##
def test_mecabstr_list():
   test = "助動詞,*,*,*,特殊・タイ,基本形,たい,タイ,タイ"
   ans = ["助動詞","*","*","*","特殊・タイ","基本形","たい","タイ","タイ"]
   assert ans == mecabstr_list(test)

def test_wakati():
    assert "なに も 考え ない で 自然 言語 処理 し たい \n" == wakati("なにも考えないで自然言語処理したい")

def test_wakati_withput_pp():
    assert "なに 考える ない 助詞 抜く たい \n" == wakati_without_pp("なにも考えないで助詞を抜きたい")

def test_tfidf():
    curpus = [
           "I want to do natural language processing without thinking about it.",
           "I want to do competition programming without thinking about it.",
           "I want to do an image search without thinking about it."
    ]
    feature_name_ans = ['about', 'an', 'competition', 'do', 'image', 'it', 'language', 'natural', 'processing', 'programming', 'search', 'thinking', 'to', 'want', 'without']
    matrix_ans       = np.array([
    [0.25318288, 0., 0., 0.25318288, 0.,0.25318288, 0.42867587, 0.42867587, 0.42867587, 0.,0., 0.25318288, 0.25318288, 0.25318288, 0.25318288],
    [0.28023746,0.,0.47448327,0.28023746,0.,0.28023746,0.,0.,0.,0.47448327,0.,0.28023746,0.28023746,0.28023746,0.28023746],
    [0.25318288,0.42867587,0.,0.25318288,0.42867587,0.25318288,0.,0.,0.,0.,0.42867587,0.25318288,0.25318288,0.25318288,0.25318288]])
    feature_name_test,matrix_test = tfidf(curpus)
    assert feature_name_test == feature_name_ans
    assert (matrix_ans == np.round(matrix_test,decimals=8)).all()

