#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import MeCab
import pytest

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
        print(list_node[6])
    return ans_text + "\n"


def tfidf(curpus: list) -> (list,np.ndarray):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(curpus)
    return vectorizer.get_feature_names(), X.toarray()

curpus = ["I want to do natural language processing without thinking about it.",
          "I want to do competition programming without thinking about it.",
          "I want to do an image search without thinking about it."
]
print(tfidf(curpus))
##
##            テスト
##
def test_mecabstr_list():
   test = "助動詞,*,*,*,特殊・タイ,基本形,たい,タイ,タイ"
   ans = ["助動詞","*","*","*","特殊・タイ","基本形","たい","タイ","タイ"]
   print(ans)
   print(mecabstr_list(test)[0])
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
    matrix_ans       = np.array([[0.25318288, 0., 0., 0.25318288, 0.,0.25318288, 0.42867587, 0.42867587, 0.42867587, 0.,0., 0.25318288, 0.25318288, 0.25318288, 0.25318288],[0.28023746,0.,0.47448327,0.28023746,0.,0.28023746,0.,0.,0.,0.47448327,0.,0.28023746,0.28023746,0.28023746,0.28023746],[0.25318288,0.42867587,0.,0.25318288,0.42867587,0.25318288,0.,0.,0.,0.,0.42867587,0.25318288,0.25318288,0.25318288,0.25318288]])
    feature_name_test,matrix_test = tfidf(curpus)
    assert feature_name_test == feature_name_ans
    assert (matrix_test == matrix_ans).all()
