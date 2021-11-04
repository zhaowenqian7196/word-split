import collections
import nltk.stem
import re
import random
import os
from nltk.tokenize import sent_tokenize
import string
from langdetect import detect


def generate_dict():
    """
    创建词典
    :return:
    """
    i = 0
    vocab = set([line.strip() for line in open('dict3.txt', encoding='utf-8')])
    vocab2 = set([line.strip() for line in open('dict4.txt', encoding='utf-8')])
    vocab3 = set.union(vocab, vocab2)

    word2idx = collections.defaultdict(int)

    with open('dict/dict6.txt', 'w', encoding='utf-8') as fd:
        j = 0
        for text in vocab3:
            lower = text.lower()
            # remove = str.maketrans('', '', string.punctuation)
            # without_punctuation = lower.translate(remove)
            words = nltk.word_tokenize(lower)
            # words = text.rstrip().split(" ")
            for word in words:
                if word not in word2idx and word not in vocab:
                    word2idx[word] = i
                    word = word.lower() + "\n"
                    fd.write(word)
                    i += 1
            j += 1


def generate_dict2():
    vocab = set([line.strip() for line in open('dict3.txt', encoding='utf-8')])
    vocab2 = set([line.strip() for line in open('dict4.txt', encoding='utf-8')])
    vocab3 = set.union(vocab, vocab2)

    with open('dict/dict6.txt', 'w', encoding='utf-8') as fd:
        for word in vocab3:
            fd.write(word + "\n")


def generate_testdata():
    i = 0
    with open('testdata.txt', 'r', encoding='utf-8') as fr, open('testdata4.txt', 'w',
                                                                 encoding='utf-8') as fd:
        for text in fr.readlines():
            newline = ''
            j = 0
            r = random.randint(9, 15)
            for word in text:
                if j == r and word != " ":
                    newline += " " + word
                else:
                    newline += word
                j += 1
            fd.write(newline)
            i += 1


def get_items():
    with open('D:/split/2021-07-26.txt', 'r', encoding='utf-8') as fr, open('text2.txt', 'w',
                                                                            encoding='utf-8') as fd:
        for text in fr.readlines():
            items = text.rstrip().split('\t')
            fd.write(items[0] + "\n")


def get_data():
    j = 0
    with open('D:/enwiki.txt', 'r', encoding='utf-8') as fr, open(
            'D:/enwiki300w.txt', 'w',
            encoding='utf-8') as fd:
        for text in fr.readlines():
            sen = list(sent_tokenize(text))
            if j >= 3000000:
                break
            try:
                for s in sen:
                    if s and detect(s) != 'en':
                        continue
                    else:
                        fd.write(s + '\n')
                        j += 1
            except:
                print(1)


def get_corpus(path):
    sentences = []
    category = os.listdir(path)
    for cg in category:
        files = os.listdir(path + '/' + cg)
        for f in files:
            if cg == "sport" and f == "199.txt":
                continue
            for line in open(path + '/' + cg + '/' + f, 'r', encoding='UTF-8'):
                if line != '\n':
                    sen = list(sent_tokenize(line))
                    for s in sen:
                        sentences.append(s)
    return sentences


def replace_func():
    with open('testdata/news2018-30w.txt', 'r', encoding='utf-8') as fr1, open(
            'testdata/news2018-30w-2.txt', 'r', encoding='utf-8') as fr2, open('testdata/news2018-60w.txt', 'w',
                                                                               encoding='utf-8') as fd:
        for line in fr1.readlines():
            line = re.sub("[{\u4e00-\u9fa5}’，。?★、…【】《》？“”‘’！]", " ", line)
            line = re.sub("https?://[A-Za-z0-9./]", " ", line)
            fd.write(line)
        for line in fr2.readlines():
            line = re.sub("[{\u4e00-\u9fa5}’，。?★、…【】《》？“”‘’！]", " ", line)
            line = re.sub("https?://[A-Za-z0-9./]", " ", line)
            fd.write(line)
        fd.close()


get_data()
