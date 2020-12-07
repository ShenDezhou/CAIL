import os
import random
import re

import pandas

dic=[]
with open('../datagen/amount_v2.dic','r',encoding='utf-8') as f:
    lines = f.readlines()
    dic.extend([l.strip() for l in lines])

print(len(dic))
types = [1] * 62 + [2] * 67
print(len(types))
regdic = []
for word in dic:
    if '_' in word:
        word = word.replace('_',r'[_\d]?',10**10)
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
            return types[i], regdic[i], dic[i]
    return None

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
category_size = 600

df = pandas.read_csv('../datagen/contract.dic',sep=',',names=['word','frequency'])
chinese_dic = df['word'].to_list()

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


tpl_amounts = []
tpl_amounts.append('218399')
tpl_amounts.append('贰拾壹万捌仟叁佰玖十九')
tpl_amounts.append('31287')
tpl_amounts.append('叁万壹仟贰佰捌拾柒')
tpl_amounts.append('300000')
tpl_amounts.append('叁拾万')
tpl_amounts.append('280000')
tpl_amounts.append('贰拾捌万')
tpl_amounts.append('135998')
tpl_amounts.append('拾叁万伍仟玖佰玖拾捌')
tpl_amounts.append('75382')
tpl_amounts.append('柒万伍仟叁佰捌拾贰')
tpl_amounts.append('135998')
tpl_amounts.append('柒万伍仟叁佰捌拾贰')
tpl_amounts.append('1800000')
tpl_amounts.append('壹佰捌拾万')

digits = list("0123456789")
formals = list("零壹贰叁肆伍陆柒捌玖")
def gen_by_tpl(seed):
    trans = ""
    for c in seed:
        if c in digits:
            trans += digits[random.randint(0, 9)]
        elif c in formals:
            trans += formals[random.randint(0, 9)]
        else:
            trans += c
    return trans


df = pandas.DataFrame(columns=["type","content"])

for dirpath, dnames, fnames in os.walk("../datagen/txt/"):
    for file in fnames:
        filename = os.path.join(dirpath, file)
        for _ in range(category_size):
            with open(filename, 'r', encoding='utf-8') as fr:
                buffer = []
                buffer_type = []
                gen_amounts = []
                for line in fr:
                    res = reg_trigger(line)
                    if res:
                        if random.randint(0, 1) == 0:
                            randstr = tpl_amounts[random.randint(0, len(tpl_amounts)-1)]
                            randstr = gen_by_tpl(randstr)
                            if random.randint(0, 1) == 0:
                                gen_amounts.append("人民币"+randstr)
                            else:
                                gen_amounts.append(randstr)
                            newline = re.sub(res[1], line, res[2] + randstr)
                        else:
                            formald = gen_with_rule(type=random.randint(1,3))
                            if random.randint(0, 1) == 0:
                                formald = "人民币" + formald
                            mathd = gen_with_rule(type=0)
                            if random.randint(0, 1) == 0:
                                mathd = "人民币" + mathd
                            gen_amounts.append(formald)
                            gen_amounts.append(mathd)

                            if '{.underline}' in line:
                                if random.randint(0, 9) <= 8:
                                    newline = re.sub('{.underline}', line, res[2] + formald + ("".join(random.sample(chinese_dic, random.randint(min(3, len(line) // 3), len(line) // 3))).replace("\n","",10 ** 10)) + mathd)
                                    buffer.append(newline.replace("\n", "", 10 ** 10))
                                else:
                                    newline = re.sub('{.underline}', line, res[2] + formald + ("".join(random.sample(chinese_dic,random.randint(min(3, len(line) // 3),len(line) // 3)))) + mathd)
                                    buffer.append(newline)
                            else:
                                if random.randint(0, 9) <= 8:
                                    newline = re.sub(res[1], line, res[2] + formald + ("".join(random.sample(chinese_dic, random.randint(min(3, len(line)//3), len(line)//3))).replace("\n", "", 10 ** 10)) +mathd)
                                    buffer.append(newline.replace("\n", "", 10 ** 10))
                                else:
                                    newline = re.sub(res[1], line, res[2] + formald + ("".join(random.sample(chinese_dic,random.randint(min(3, len(line) // 3),len(line) // 3)))) + mathd)
                                    buffer.append(newline)
                        buffer_type.append(res[0])
                    else:
                        buffer.append(line.strip())
                        buffer_type.append(0)

                sent_dic = dict(zip(buffer,buffer_type))
                for i, item in sent_dic.items():
                    df = df.append({"type": i, "content": item}, ignore_index=True)
                # contracts = r"\n".join(buffer)
                # contracts = contracts.replace("\r\n", r"\n", 10 ** 10)
                # contracts = contracts.replace("\n", r"\n", 10 ** 10)
                # contracts = contracts.replace("{.underline}", "", 10 ** 10)
                # index_marks = []
                # for amount in gen_amounts:
                #     start = contracts.find(amount)
                #     end = start + len(amount)
                #     if start>0:
                #         index_marks.append((start,end))
                # if index_marks:
                #     indexs =";".join( [str(k)+','+str(v) for (k,v) in index_marks] )

df.to_csv("contract_type_train.csv", columns=["type", "content"], index=False)
print('FIN')