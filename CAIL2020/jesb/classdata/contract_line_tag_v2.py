import os
import random
import re

import pandas

dic1=[]
with open('amount.dic','r',encoding='utf-8') as f:
    lines = f.readlines()
    dic1.extend([l.strip() for l in lines])

dic2=[]
with open('amount_v2.dic','r',encoding='utf-8') as f:
    lines = f.readlines()
    dic2.extend([l.strip() for l in lines])

#
# print(len(dic))
# types = [1] * 62 + [2] * 67
# print(len(types))
# regdic = []
# for word in dic:
#     if '_' in word:
#         word = word.replace('_',r'\s?\d?',10**10)
#         r = re.compile(word)
#     else:
#         r = re.compile(word)
#     regdic.append(r)

pattern_all_dic = ""
for line in dic1 + dic2:
    pattern_all_dic += "(%s)|" % line.strip()
pattern_all_dic = pattern_all_dic.strip('|')
pattern_all = re.compile(pattern_all_dic)

pattern_dic_v2 = ""
with open('amount_v2.dic','r',encoding='utf-8') as f:
    for line in f:
        line = line.strip().replace('___','\d{1,6}',10**10)
        pattern_dic_v2 += "(%s)|" % line.strip()
    pattern_dic_v2 = pattern_dic_v2.strip('|')
pattern2 = re.compile(pattern_dic_v2)


pattern_dic = ""
with open('amount.dic','r',encoding='utf-8') as f:
    for line in f:
        pattern_dic += "(%s)|" % line.strip()
    pattern_dic = pattern_dic.strip('|')
pattern1 = re.compile(pattern_dic)

pattern_dic_v2 = ""
with open('amount_v2.dic','r',encoding='utf-8') as f:
    for line in f:
        line = line.strip().replace('_','\d{0,6}',10**10)
        pattern_dic_v2 += "(%s)|" % line.strip()
    pattern_dic_v2 = pattern_dic_v2.strip('|')
pattern2 = re.compile(pattern_dic_v2)


def reg_trigger(line):
    res = pattern_all.match(line)
    if res:
        res = pattern1.match(line)
        if res:
            return 1, line
        res = pattern2.match(line)
        if res:
            return 2, line
    return 0, line

FORMAL_DIGIT="零一二三四五六七八九十百千万亿"
LARGE_FORMAL_DIGIT="零壹贰叁肆伍陆柒捌玖拾佰仟萬億"
DIGIT_PAUSE=',\uFF0C'
DIGIT_SPLIT='.'
CARRY = "十百千万亿"
LARGE_CARRY = "拾佰仟萬億"
COMMON_UNIT='元'
UNIT='元角分'
math_digit="1234567890\uFF10\uFF11\uFF12\uFF13\uFF14\uFF15\uFF16\uFF17\uFF18\uFF19"

amount_len = (1, 16)
category_size = 1

df = pandas.read_csv('../datagen/contract.dic',sep=',',names=['word','frequency'])
chinese_dic = df['word'].to_list()

df = pandas.read_csv('../data/contract_amount_train_v6.csv')
df_result = pandas.DataFrame()
noise = []
for index, row in df.iterrows():
    lines = re.sub(r'\n+',r'\n', row[0]).split(r'\n')
    for line in lines:
        type, l = reg_trigger(line)
        if len(l) < 10:
            continue
        l = l.replace("\n",r"\n",10**10)
        if type:
            df_result = df_result.append({"type":int(type), "content": l}, ignore_index=True)
        else:
            noise.append(l)

noise = [n for n in noise if len(n) > 20]
total_len = len(df_result) // 2
count = 0
for i in range(len(noise)):
    df_result = df_result.append({"type":0, "content": noise[i]}, ignore_index=True)
    count += 1
    if count > total_len:
        break
df_result['type'] = df_result['type'].astype(int)
df_result.to_csv("../data/contract_type_train_v2.csv", columns=['type','content'], index=False)
print('FIN')