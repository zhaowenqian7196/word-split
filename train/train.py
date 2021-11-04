import pickle
from nltk.corpus import reuters
import os
import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize

term_count, bigram_count = {}, {}


def get_pdic(corpus):
    for line in corpus:
        for i in range(len(line)):
            term = line[i]
            bigram = ''.join(line[i:i + 2]) if i < len(line) - 1 else ''
            term_count[term] = term_count[term] + 1 if term in term_count else 1
            if bigram:
                bigram_count[bigram] = bigram_count[bigram] + 1 if bigram in bigram_count else 1
    return term_count, bigram_count


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def preprocessing(article):
    """
    语料的清洗
    :param article:
    :return:
    """
    article = article.str.lower()
    article = article.apply(lambda x: cleanhtml(x))
    article = article.apply(lambda x: re.sub('\S+@\S+', '', x))
    article = article.apply(lambda x: re.sub(
        "((http\://|https\://|ftp\://)|(www.))+(([a-zA-Z0-9\.-]+\.[a-zA-Z]{2,4})|([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}))(/[a-zA-Z0-9%:/-_\?\.'~]*)?",
        '', x))
    article = article.apply(lambda x: x.replace("\xa0", " "))
    article = article.apply(lambda x: x.replace('‘', "'"))
    article = article.apply(lambda x: x.replace('’', "'"))
    article = article.apply(lambda x: x.replace('”', '"'))
    article = article.apply(lambda x: x.replace('“', '"'))
    article = article.apply(lambda x: re.sub(' +', ' ', x))
    article = article.apply(lambda x: re.sub(' +', ' ', x))
    return article


def getcsv_corpus(path):
    sentences = []
    df = pd.read_csv(path)
    article = df['content']
    article = preprocessing(article)
    sen = article.values

    for s in sen:
        sen = list(sent_tokenize(s))
        for s in sen:
            sentences.append(s)
    return sentences


def get_data(text_file):
    """
    获取数据
    :param text_file:
    :return:
    """
    data = []
    with open(text_file, 'r', encoding='utf-8') as fr:
        for text in fr.readlines():
            words = nltk.word_tokenize(text)
            data.append(words)
    return data


def get_corpus(path):
    sentences = []
    category = os.listdir(path)
    for cg in category:
        files = os.listdir(path + '/' + cg)
        for f in files:
            try:
                for line in open(path + '/' + cg + '/' + f, 'r', encoding='UTF-8'):
                    if line != '\n':
                        sen = list(sent_tokenize(line))
                        for s in sen:
                            sentences.append(s)
            except:
                print()
    return sentences


def main():
    categories = reuters.categories()
    corpus = reuters.sents(categories=categories)
    get_pdic(corpus)

    train_sentences1 = get_corpus('./corpus/News Articles')
    train_sentences2 = getcsv_corpus('./corpus/articles1.csv')
    train_sentences = train_sentences1 + train_sentences2
    tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in train_sentences]

    get_pdic(tokenized_text)
    bigram = [term_count, bigram_count]
    biGramDictFile = open("bigram.pkl", 'wb')
    pickle.dump(bigram, biGramDictFile, 0)


if __name__ == '__main__':
    main()
