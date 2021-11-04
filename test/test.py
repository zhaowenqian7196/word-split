import pickle
import argparse
import numpy as np
import re
import nltk


def generate_candidates(index, line):
    """
    生成不同拼接距离的候选词
    :param index:
    :param line:
    :return:
    """
    dis = 3
    candidates = []
    i = 1
    while i <= dis:
        for j in range(i + 1):
            if index - j < 0 or index + i - j >= len(line):
                continue
            candi = [''.join(line[index - j:index + i - j + 1]), index - j, index + i - j]
            candidates.append(candi)
        i += 1
    return [word for word in candidates if word[0] in vocab]


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


def cal_ppl(line):
    i = 1
    prob = 0
    while i < len(line):
        pre = line[i - 1]
        if pre and pre + line[i] in bigram_count and pre in term_count:
            prob += np.log((bigram_count[pre + line[i]] + 1.0) / (term_count[pre] + V))
        else:
            prob += np.log(1.0 / V)
        i += 1
    return pow(2, -prob / len(line))


def word_joint(line):
    """
    处理句子
    :param line: 需要处理的当前句子
    :return: 拼接之后的句子
    """
    threshold = 0.1
    term = 0
    newline = [line[0]]
    pre = cal_ppl(line)
    i = 1
    while i < len(line):
        match = re.match(r'[A-Za-z]', line[i])
        if not match:
            newline.append(line[i])
            i += 1
            continue
        pre_word = newline[len(newline) - 1]
        w = pre_word + line[i]
        if w not in bigram_count:
            term += 1
        if term >= 1:
            candidates = generate_candidates(i, line)
            ppls = [pre]
            min_ppl, ppls = get_min_ppl(ppls, line, newline, candidates, i)

            if candidates and (pre - min_ppl) / pre > threshold:
                min_idx = ppls.index(min(ppls)) - 1
                pre = min(ppls)
                if min_idx != -1:
                    min_start = candidates[min_idx][1]
                    min_end = candidates[min_idx][2]
                    if min_start < i:
                        del newline[min_start - i:]
                    if min_end > i:
                        i = min_end  # 如果单词已经把后面的拼接了，就跳过
                    newline.append(candidates[min_idx][0])
                else:
                    newline.append(line[i])
            else:
                newline.append(line[i])
            term = 0
        else:
            newline.append(line[i])
        i += 1
    return newline


def get_min_ppl(ppls, line, newline, candidates, i):
    """
    找到最小的困惑度
    :param line: 需要处理的句子
    :param newline:
    :param candidates:
    :param i:
    :return:
    """
    for candi in candidates:
        new = []
        start = candi[1]
        end = candi[2]
        for j in range(len(newline)):
            new.append(newline[j])
        if start < i:
            del new[start - i:]
        new.append(candi[0])
        j = end + 1
        while j < len(line):
            new.append(line[j])
            j += 1
        ppl = cal_ppl(new)
        ppls.append(ppl)
    min_ppl = min(ppls)
    return min_ppl, ppls


biGramDictFile = open("../train/bigram.pkl", 'rb')
bigram = pickle.load(biGramDictFile)
term_count = bigram[0]
bigram_count = bigram[1]
V = len(term_count)
vocab = set([line.strip() for line in open('./dict/dict7.txt', encoding='utf-8')])


def main():
    parser = argparse.ArgumentParser(description='testfile')
    parser.add_argument('--orgs', type=str, default='./testdata/testdata.txt', help='orgs')
    parser.add_argument('--tests', type=str, default='./testdata/testdata4.txt', help='tests')
    args = parser.parse_args()

    orgs = get_data(args.orgs)
    tests = get_data(args.tests)

    index = 0
    correct = 0
    for test in tests:
        newline = word_joint(test)
        total = index + 1
        if newline == orgs[index]:
            correct += 1
        print('----Original input: %s\n----Correct input: %s' % (test, newline))
        print('----Correct: %d \nTotal: %d' % (correct, total))
        index += 1
    print('Accuracy: %f \n' % (correct / total))


if __name__ == '__main__':
    main()
