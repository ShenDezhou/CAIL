import os
import random
import re

import pandas

dic=[]
with open('amount_v1v2.dic', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    dic.extend([l.strip() for l in lines])

print(len(dic))
regdic = []
for word in dic:
    if '_' in word:
        word = word.replace('_',r'[_\d]?',10**10)
    #     r = re.compile("%s\s+(\[\s+\])" % word)
    # else:
    r = re.compile(word + ".+(\[\s+\])")
    regdic.append(r)
    r = re.compile(word + ".+({.underline})")
    regdic.append(r)
    r = re.compile(word + ".+([▁_＿]+)")
    regdic.append(r)


def trigger(line):
    for i in range(len(dic)):
        if dic[i] in line:
            return i, dic[i]
    return None

def reg_trigger(line):
    for i in range(len(regdic)):
        match = regdic[i].search(line)
        if match:
            return i, match.group(1), dic[i//3], match.group(1)

    # match = re.compile("(人民币.+元)").search(line)
    # if match:
    #     return 1, match.group(1), "", match.group(1)
    # match = re.compile("大写：(.+元)整").search(line)
    # if match:
    #     return 1, match.group(1), "", match.group(1)
    match = re.compile("（[¥￥]([\u3000\s]+)）").search(line)
    if match:
        return 0, match.group(1), "", match.group(1)
    return None


def format2digit(self, word):
    trans = ""
    if word.startswith('十'):
        trans += '1'

    for c in word:
        if c in self.FORMAL_DIGIT:
            trans += self.math_digit[self.FORMAL_DIGIT.index(c)]
        if c == '千' and not word.endswith('千'):
            if '百' not in word and '十' not in word:
                trans += "0"
        if word.endswith(c):
            if c == "十":
                trans += '0'
            if c == "百":
                trans += '00'
            if c == "千":
                trans += '000'
    return trans

FORMAL_DIGIT="零一二三四五六七八九"
LARGE_FORMAL_DIGIT="零壹贰叁肆伍陆柒捌玖"
DIGIT_PAUSE=',\uFF0C'
DIGIT_SPLIT='.'
CARRY = "十百千万亿"
LARGE_CARRY = "拾佰仟萬億"
PREFIX="人民币"
PREFIX_SIGN="¥￥"
COMMON_UNIT='元'
UNIT='元角分'
math_digit="1234567890"
full_math_digit="\uFF10\uFF11\uFF12\uFF13\uFF14\uFF15\uFF16\uFF17\uFF18\uFF19"


def digit2formal(word):
    trans = ""

    natural = True
    fraction = 0
    for index, c in enumerate(str(word)):
        if c == ".":
            natural = False
            continue
        if natural:
            if index % 4 == 1:
                trans += '拾'
            elif index % 4 == 2:
                trans += '佰'
            elif index % 4 == 3:
                trans += '仟'
            elif index == 4:
                trans += '万'
            elif index == 8:
                trans += '亿'
            if len(trans)> 1 and trans[-1] == '零' and LARGE_FORMAL_DIGIT[int(c)] =='零':
                continue
            trans += LARGE_FORMAL_DIGIT[int(c)]
        else:
            if fraction == 0:
                trans = trans[::-1]
                trans += "元" + LARGE_FORMAL_DIGIT[int(c)] + "角"
            elif fraction == 1:
                trans += LARGE_FORMAL_DIGIT[int(c)] + '分'
            fraction += 1
    if natural:
        trans = trans[::-1]
    return trans



amount_len = (1, 16)
category_size = 1000

df = pandas.read_csv('contract.dic',sep=',',names=['word','frequency'])
chinese_dic = df['word'].to_list()

def gen_amount(formal = 0):
    len = random.randint(*amount_len)
    if formal==0:
        return "".join(random.sample(list(math_digit), len))
    elif formal == 1:
        return "".join(random.sample(list(FORMAL_DIGIT), len))
    elif formal == 2:
        return "".join(random.sample(list(full_math_digit), len))
    elif formal == 3:
        return "".join(random.sample(list(math_digit) + list(full_math_digit), len))
    return "".join(random.sample(list(LARGE_FORMAL_DIGIT), len))


def gen_digit_amount(lenghth, comma=True, period=True):
    index = 0
    gen = ""
    dice = random.random()
    while index < lenghth:
        if dice < 0.7:
            gen += "".join(random.sample(list(math_digit), 3))
        elif dice < 0.9:
            gen += "".join(random.sample(list(full_math_digit), 3))
        else:
            gen += "".join(random.sample(list(math_digit) + list(full_math_digit), 3))
        if comma:
            if dice < 0.7:
                gen += DIGIT_PAUSE[0]
            elif dice < 0.9:
                gen += DIGIT_PAUSE[1]
            else:
                gen += "".join(random.sample(list(DIGIT_PAUSE), 1))
        index += 3
    gen = gen.strip(",")
    gen = gen.strip("\uFF0C")

    if period:
        gen += "."
        if dice < 0.7:
            gen += "".join(random.sample(list(math_digit), 2))
        elif dice < 0.9:
            gen += "".join(random.sample(list(full_math_digit), 2))
        else:
            gen += "".join(random.sample(list(math_digit) + list(full_math_digit), 2))
    return gen

def gen_with_rule(type = 0):
    dice = random.random()
    if dice < 0.4:
        chars = random.randint(1, 10)
    elif dice < 0.8:
        chars = random.randint(1, 6)
    else:
        chars = random.randint(1, 3)
    index = 0
    gen = ""

    #数字
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
        gen = gen.lstrip('0')
        gen = gen.lstrip('０')

    else:
        _dice = random.random()

        if _dice < 0.5:
            digits = gen_digit_amount(chars, comma=False, period=False)
            gen = digit2formal(digits)
        elif _dice < 0.99:
            digits = gen_digit_amount(chars, comma=False, period=True)
            gen = digit2formal(digits)
        else:
            # 大写
            while index < chars:
                if type == 1:
                    gen += "".join(random.sample(list(FORMAL_DIGIT), 1))
                    gen += "".join(random.sample(list(CARRY), 1))
                    index += 2
                elif type == 2:
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
            if type == 0:
                gen += UNIT[0]
            else:
                if random.random() < 0.1:
                    gen += UNIT[0] + random.sample(list(FORMAL_DIGIT), 1)[0] + UNIT[1]
                else:
                    gen += UNIT[0] + random.sample(list(FORMAL_DIGIT), 1)[0] + UNIT[1] + \
                           random.sample(list(FORMAL_DIGIT), 1)[0] + UNIT[2]
            gen = gen.lstrip("零")

    gen = gen.strip(",")
    gen = gen.strip("\uFF0C")

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

def clean_line(contracts):
    contracts = contracts.replace("*", "", 10 ** 10)
    contracts = contracts.replace("#", "", 10 ** 10)
    contracts = contracts.replace("\\", "", 10 ** 10)
    contracts = contracts.replace(">", "", 10 ** 10)
    # contracts = contracts.replace("{.underline}", "", 10 ** 10)
    return contracts


df = pandas.DataFrame(columns=["contract","indexes"])

for dirpath, dnames, fnames in os.walk("txt/"):
    for file in fnames:
        filename = os.path.join(dirpath, file)
        for _ in range(category_size):
            with open(filename, 'r', encoding='utf-8') as fr:
                buffer = []
                gen_amounts = []
                for line in fr:
                    line = clean_line(line)
                    res = reg_trigger(line)
                    if res:
                        dice = random.randint(0, 100)
                        if dice == 0:
                            randstr = tpl_amounts[random.randint(0, len(tpl_amounts)-1)]
                            randstr = gen_by_tpl(randstr)
                            if random.randint(0, 1) == 0:
                                gen_amounts.append("人民币"+randstr)
                            else: #¥￥
                                gen_amounts.append(random.sample(list(PREFIX_SIGN), 1)[0]+randstr)
                            newline = re.sub(res[1], randstr, line)
                            buffer.append(newline)
                        else:
                            start_index = line.find(res[2]) + len(res[2])

                            if '¥' in line or '￥' in line or '大写' in line:
                                if '大写' in line:
                                    start_index = line.find('大写') + 2
                                    formald = gen_with_rule(type=2)
                                    if random.randint(0, 1) == 0:
                                        formald = "人民币" + formald
                                    gen_amounts.append(formald)
                                    newline = line[:start_index] + re.sub(res[3], formald, line[start_index:], count=1)
                                elif '￥' in line:
                                    start_index = line.find('￥') + 1
                                    digitid = gen_with_rule(type=0)
                                    if random.randint(0, 1) == 0:
                                        digitid = '￥' + digitid
                                    gen_amounts.append(digitid)
                                    newline = line[:start_index] + re.sub(res[3],  digitid, line[start_index:], count=1)
                                elif '¥' in line:
                                    start_index = line.find('¥') + 1
                                    digitid = gen_with_rule(type=0)
                                    if random.randint(0, 1) == 0:
                                        digitid = '¥' + digitid
                                    gen_amounts.append(digitid)
                                    newline = line[:start_index] + re.sub(res[3],  digitid, line[start_index:], count=1)
                                else:
                                    gen_type = random.randint(0,1)
                                    mis_formald = gen_with_rule(type=gen_type)
                                    if gen_type == 1:
                                        mis_formald = "人民币" + mis_formald
                                    else:
                                        mis_formald = random.sample(list(PREFIX_SIGN), 1)[0] + mis_formald +"元"
                                    gen_amounts.append(mis_formald)
                                    newline = line[:start_index] + re.sub(res[3],   mis_formald, line[start_index:], count=1)
                                buffer.append(newline)
                            else:
                                if random.randint(0, 1) == 0:
                                    formald = gen_with_rule(type=2)
                                    formald = "人民币" + formald
                                    gen_amounts.append(formald)
                                    newline = line[:start_index] + re.sub(res[3],  formald,  line[start_index:], count=1)
                                else:
                                    digitid = gen_with_rule(type=0)
                                    digitid = random.sample(list(PREFIX_SIGN), 1)[0] + digitid
                                    gen_amounts.append(digitid)
                                    newline = line[:start_index] + re.sub(res[3],   digitid,  line[start_index:], count=1)
                                buffer.append(newline)
                    else:
                        buffer.append(line.strip())

                contracts = r"\n".join(buffer)

                contracts = contracts.replace("[", "", 10 ** 10)
                contracts = contracts.replace("]", "", 10 ** 10)
                contracts = contracts.replace("/", "", 10 ** 10)

                contracts = contracts.replace("▁", "", 10 ** 10)
                contracts = contracts.replace("_", "", 10 ** 10)
                contracts = contracts.replace("＿", "", 10 ** 10)

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
df.to_csv("../data/contract_amount_train_v6.csv", columns=["contract", "indexes"], index=False)
print('FIN')