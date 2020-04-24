#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import MeCab
import sklearn
import pytest

def wakati(text: str) -> str:
    return MeCab.Tagger("-Owakati").parse(text)

## pp = postpositional particle(助詞)
def wakati_without_pp(text: str) -> list:
    tagger = MeCab.Tagger("-Ochasen")
    tagger.parse("")
    node = tagger.parseToNode(text)
    word_class = []
    while node:
        word = node.surface
        wclass = node.feature.split(',')
        if wclass[0] != u'BOS/EOS':
            if wclass[6] == None:
                word_class.append((word,wclass[0],wclass[1],wclass[2],""))
            else:
                word_class.append((word,wclass[0],wclass[1],wclass[2],wclass[6]))
        node = node.next
    return word_class


def test_wakati():
    assert "なに も 考え ない で 自然 言語 処理 し たい \n" == wakati("なにも考えないで自然言語処理したい")

def test_wakati_withput_pp():
    assert "なに 考え ない 助詞 抜き たい \n" == wakati_without_pp("なにも考えないで助詞を抜きたい") 
