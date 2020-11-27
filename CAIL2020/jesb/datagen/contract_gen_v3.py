import os
import random
import re

import pandas

dic=[]
with open('amount_v2.dic','r',encoding='utf-8') as f:
    lines = f.readlines()
    dic.extend([l.strip() for l in lines])

print(len(dic))
regdic = []
for word in dic:
    if '_' in word:
        word = word.replace('_','[_\d]?',10)
        r = re.compile(word)
    else:
        r = re.compile(word)
    regdic.append(r)

def trigger(line):
    for i in range(len(dic)):
        if dic[i] in line:
            return i, dic[i]
    return None

def reg_trigger(line):
    for i in range(len(regdic)):
        if regdic[i].search(line):
            return i, regdic[i], dic[i]
    return None

FORMAL_DIGIT="零一二三四五六七八九十百千万亿兆"
LARGE_FORMAL_DIGIT="零壹贰叁肆伍陆柒捌玖拾佰仟萬億"
DIGIT_PAUSE=',\uFF0C'
DIGIT_SPLIT='.'
CARRY = "十百千万亿"
LARGE_CARRY = "拾佰仟萬億"
COMMON_UNIT='元'
UNIT='元角分'
math_digit="1234567890\uFF10\uFF11\uFF12\uFF13\uFF14\uFF15\uFF16\uFF17\uFF18\uFF19"

amount_len = (1, 5)
category_size = 400

chinese_dic = []
with open("chinese4049.txt",'r',encoding='utf-8') as f:
    for line in f:
        chinese_dic.append(line.strip())

def gen_amount(formal = 0):
    len = random.randint(*amount_len)
    if formal==0:
        return "".join(random.sample(list(math_digit), len))
    elif formal == 1:
        return "".join(random.sample(list(FORMAL_DIGIT), len))
    return "".join(random.sample(list(LARGE_FORMAL_DIGIT), len))


def gen_digit_amount(lenghth, comma=True, period=True):
    index = 0
    gen = ""
    while index < lenghth:
        gen += "".join(random.sample(list(math_digit), 3))
        if comma:
            gen += "".join(random.sample(list(DIGIT_PAUSE), 1))
        index += 3
    if period:
        gen += "".join(random.sample(list(math_digit), 2))
    return gen

def gen_with_rule(type = 0):
    dice = random.random()
    if dice < 0.3:
        chars = random.randint(1, 3)
    elif dice<0.6:
        chars = random.randint(1, 6)
    else:
        chars = random.randint(1, 10)
    index = 0
    gen = ""

    if type==0:
        digit_type = random.randint(1, 4)
        if digit_type ==0:
            gen = gen_digit_amount(chars, comma=False, period=False)
        elif digit_type ==1:
            gen = gen_digit_amount(chars, comma=False, period=True)
        elif digit_type == 2:
            gen = gen_digit_amount(chars, comma=True, period=False)
        else:
            gen = gen_digit_amount(chars, comma=True, period=True)
    else:
        while index < chars:
            if type==1:
                gen += "".join(random.sample(list(FORMAL_DIGIT), 1))
                gen += "".join(random.sample(list(CARRY), 1))
                index+=2
            elif type==2:
                gen += "".join(random.sample(list(LARGE_FORMAL_DIGIT), 1))
                gen += "".join(random.sample(list(LARGE_CARRY), 1))
                index += 2
            else:
                if random.random() > 0.5:
                    gen += "".join(random.sample(list(math_digit), 3))
                    if random.random() > 0.1:
                        gen += "".join(random.sample(list(DIGIT_PAUSE), 1))
                    if random.random() > 0.5:
                        gen += "".join(random.sample(list(CARRY), 1))
                    else:
                        gen += "".join(random.sample(list(LARGE_CARRY), 1))
                    index += 4
                else:
                    gen += "".join(random.sample(list(FORMAL_DIGIT), 1))
                    gen += "".join(random.sample(list(LARGE_FORMAL_DIGIT), 1))
                    index += 2
    if random.random() < 0.1:
        gen += "".join(random.sample(list(UNIT), 1))
    elif random.random() < 0.9:
        gen += COMMON_UNIT
    else:
        gen += " "
    return gen



df = pandas.DataFrame(columns=["contract","indexes"])

for dirpath, dnames, fnames in os.walk("txt/"):
    for file in fnames:
        filename = os.path.join(dirpath, file)
        for _ in range(category_size):
            with open(filename, 'r', encoding='utf-8') as fr:
                buffer = []
                gen_amounts = []
                for line in fr:
                    res = reg_trigger(line)
                    if res:
                        formald = gen_with_rule(type=random.randint(1,3))
                        mathd = gen_with_rule(type=0)
                        gen_amounts.append(formald)
                        gen_amounts.append(mathd)
                        newline = res[1].sub(line, res[2] + formald + ("".join(random.sample(chinese_dic, random.randint(min(3, len(line)//3), len(line)//3))).replace("\n", "", 10 ** 10)) +mathd)
                        buffer.append(newline.strip())
                    else:
                        buffer.append(line.strip())

                contracts = r"\n".join(buffer)
                contracts = contracts.replace("\r\n", r"\n", 10 ** 10)
                contracts = contracts.replace("\n", r"\n", 10 ** 10)
                contracts = contracts.replace("{.underline}", "", 10 ** 10)
                index_marks = []
                for amount in gen_amounts:
                    start = contracts.find(amount)
                    end = start + len(amount)
                    if start>0:
                        index_marks.append((start,end))
                if index_marks:
                    indexs =";".join( [str(k)+','+str(v) for (k,v) in index_marks] )
                    df = df.append({"contract":contracts,"indexes": indexs}, ignore_index=True)
df.to_csv("contract_amount_train_v3.csv", columns=["contract", "indexes"], index=False)
print('FIN')