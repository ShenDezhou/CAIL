import itertools
import json
import os
import random
import re

import pandas

totaldic = []
dic=[]
with open('amount.dic', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    dic.extend([l.strip() for l in lines if len(l.strip())])
    totaldic.extend(dic)

dic2=[]
with open('amount_2.dic', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    dic2.extend([l for l in lines if len(l.strip())])
    # dic2 = [l for l in dic2 if len(l)]
    # dic2 = list(itertools.chain(*dic2))
    totaldic.extend(dic2)


dic3=[]
with open('amount_3.dic', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    dic3.extend([l.strip() for l in lines if len(l.strip())])
    totaldic.extend(dic3)


def getType(line):
    result = 0
    result2 = 0
    result3 = 0
    for k in dic:
        r = re.compile(k)
        l = r.findall(line)
        result += len(l)

    for k in dic2:
        r = re.compile(k)
        l = r.findall(line)
        result2 += len(l)

    for k in dic3:
        r = re.compile(k)
        l = r.findall(line)
        result3 += len(l)

    if result2>0:
        return 'PHA'

    if result>0:
        return 'TOT'

    if result3>0:
        return "RES"

    return None


print(len(totaldic))
regdic = []
for word in totaldic:
    # word = word.replace('_', r'[_\d]?', 10 ** 10)
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
            return i, match.group(1), totaldic[i//3], match.group(1)

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


def digit2formal(word, prefix_type = 0):
    trans = ""

    word = word.replace("万","").replace('元','')
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
        trans = trans[::-1] + "元"
    if prefix_type== 1:
        trans = "人民币"+ trans
    return trans



amount_len = (1, 16)
category_size = 1

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


def gen_digit_amount(lenghth, comma=True, period=True, ten_thousand=False, prefix_flag=0):
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
    gen = gen.lstrip('0')
    gen = gen.lstrip('０')

    if period:
        gen += "."
        if dice < 0.7:
            gen += "".join(random.sample(list(math_digit), 2))
        elif dice < 0.9:
            gen += "".join(random.sample(list(full_math_digit), 2))
        else:
            gen += "".join(random.sample(list(math_digit) + list(full_math_digit), 2))

    if ten_thousand:
        gen += '万元'
    else:
        gen += "元"
    if prefix_flag ==0:
        gen = gen
    elif prefix_flag == 1:
        gen = "¥"+gen
    elif prefix_flag == 2:
        gen = "￥"+gen
    elif prefix_flag == 3:
        gen = "人民币" + gen
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
    if type==0 or type==1:
        digit_type = random.randint(1, 4)
        ten_thousand = type
        prefix_type = random.randint(0,3)
        if digit_type ==0:
            gen = gen_digit_amount(chars, comma=False, period=False, ten_thousand=ten_thousand, prefix_flag=prefix_type)
        elif digit_type ==1:
            gen = gen_digit_amount(chars, comma=False, period=True, ten_thousand=ten_thousand, prefix_flag=prefix_type)
        elif digit_type == 2:
            gen = gen_digit_amount(chars, comma=True, period=False, ten_thousand=ten_thousand, prefix_flag=prefix_type)
        else:
            gen = gen_digit_amount(chars, comma=True, period=True, ten_thousand=ten_thousand, prefix_flag=prefix_type)

    else:
        _dice = random.random()

        if _dice < 0.5:
            digits = gen_digit_amount(chars, comma=False, period=False)
            prefix_type = random.randint(0,1)
            gen = digit2formal(digits,prefix_type=prefix_type)
        elif _dice < 0.99:
            digits = gen_digit_amount(chars, comma=False, period=True)
            prefix_type = random.randint(0, 1)
            gen = digit2formal(digits, prefix_type=prefix_type)
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
tpl_amounts.append('1.99万元')
tpl_amounts.append('218399元')
tpl_amounts.append('贰拾壹万捌仟叁佰玖十九元')
tpl_amounts.append('31287元')
tpl_amounts.append('叁万壹仟贰佰捌拾柒元')
tpl_amounts.append('300000元')
tpl_amounts.append('叁拾万元')
tpl_amounts.append('280000元')
tpl_amounts.append('贰拾捌万元')
tpl_amounts.append('135998元')
tpl_amounts.append('拾叁万伍仟玖佰玖拾捌元')
tpl_amounts.append('75382元')
tpl_amounts.append('柒万伍仟叁佰捌拾贰元')
tpl_amounts.append('135998元')
tpl_amounts.append('柒万伍仟叁佰捌拾贰元')
tpl_amounts.append('1800000元')
tpl_amounts.append('壹佰捌拾万元')
tpl_amounts.append('1.99万元')

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
    return contracts


def clean_line2(contracts):
    contracts = contracts.replace("万", "", 10 ** 10)
    contracts = contracts.replace("元", "", 10 ** 10)
    contracts = contracts.replace("角", "", 10 ** 10)
    contracts = contracts.replace("分", "", 10 ** 10)
    contracts = contracts.replace("\n", "", 10 ** 10)
    return contracts

def clean_line3(contracts):
    contracts = contracts.replace("[", "", 10 ** 10)
    contracts = contracts.replace("]", "", 10 ** 10)
    contracts = contracts.replace("/", "", 10 ** 10)
    contracts = contracts.replace("▁", "", 10 ** 10)
    contracts = contracts.replace("_", "", 10 ** 10)
    contracts = contracts.replace("＿", "", 10 ** 10)
    contracts = contracts.replace("¥¥", "¥", 10 ** 10)
    contracts = contracts.replace("￥￥", "￥", 10 ** 10)
    contracts = contracts.replace("¥￥", "¥", 10 ** 10)
    contracts = contracts.replace("￥¥", "￥", 10 ** 10)
    contracts = contracts.replace(" ", "", 10 ** 10)
    contracts = contracts.replace("{.underline}", "", 10 ** 10)
    return contracts

contract_list = []
# df = pandas.DataFrame(columns=["contract","indexes"])

for dirpath, dnames, fnames in os.walk("txt/"):
    for file in fnames:
        filename = os.path.join(dirpath, file)
        for _ in range(category_size):
            with open(filename, 'r', encoding='utf-8') as fr:
                buffer = []
                for paragraph in fr:
                    paragraph = clean_line(paragraph)
                    lines = paragraph.split("。")
                    for line in lines:
                        res = reg_trigger(line)
                        if res:
                            item = {}
                            gen_amounts = []
                            gen_types = []
                            dice = random.randint(0, 100)
                            if dice == 0:
                                randstr = tpl_amounts[random.randint(0, len(tpl_amounts)-1)]
                                randstr = gen_by_tpl(randstr)
                                sample_type = random.randint(0, 3)
                                if sample_type == 0:
                                    gen_amounts.append("人民币"+randstr)
                                elif sample_type == 1: #¥￥
                                    gen_amounts.append(random.sample(list(PREFIX_SIGN), 1)[0]+randstr)
                                else:
                                    gen_amounts.append(randstr)
                                newline = re.sub(res[1], randstr, line, count=1)
                                gen_types.append(getType(newline))
                                buffer.append(newline)
                            else:
                                start_index = line.find(res[2]) + len(res[2])
                                line = clean_line2(line)
                                if '¥' in line or '￥' in line or '大写' in line:
                                    for index,word in enumerate(['大写','¥','￥']):
                                        if word in line:
                                            start_index = line.find(word) + len(word)
                                            if start_index != 0:
                                                if index==0:
                                                    type = 2
                                                if index==1 or index==2:
                                                    type = random.randint(0,1)
                                                formald = gen_with_rule(type=type)
                                                # if random.randint(0, 1) == 0:
                                                #     formald = "人民币" + formald
                                                gen_amounts.append(formald)
                                                line = line[:start_index] + re.sub(res[3], formald, line[start_index:], count=1)
                                                gen_types.append(getType(line))
                                        #
                                        # elif '￥' in line:
                                        #     start_index = line.find('￥') + 1
                                        #     digitid = gen_with_rule(type=random.randint(0,1))
                                        #     # if random.randint(0, 1) == 0:
                                        #     #     digitid = '￥' + digitid
                                        #     gen_amounts.append(digitid)
                                        #     newline = line[:start_index] + re.sub(res[3],  digitid, line[start_index:], count=1)
                                        # elif '¥' in line:
                                        #     start_index = line.find('¥') + 1
                                        #     digitid = gen_with_rule(type=random.randint(0,1))
                                        #     # if random.randint(0, 1) == 0:
                                        #     #     digitid = '¥' + digitid
                                        #     gen_amounts.append(digitid)
                                        #     newline = line[:start_index] + re.sub(res[3],  digitid, line[start_index:], count=1)
                                        # else:
                                        #     gen_type = random.randint(0,2)
                                        #     mis_formald = gen_with_rule(type=gen_type)
                                        #     # if gen_type == 1:
                                        #     #     mis_formald = "人民币" + mis_formald
                                        #     # else:
                                        #     #     mis_formald = random.sample(list(PREFIX_SIGN), 1)[0] + mis_formald +"元"
                                        #     gen_amounts.append(mis_formald)
                                        #     newline = line[:start_index] + re.sub(res[3],   mis_formald, line[start_index:], count=1)
                                    newline = line
                                    buffer.append(newline)


                                else:
                                    sample_type = random.randint(0, 3)
                                    if sample_type == 0:
                                        formald = gen_with_rule(type=2)
                                        # formald = "人民币" + formald
                                        gen_amounts.append(formald)
                                        newline = line[:start_index] + re.sub(res[3],  formald,  line[start_index:], count=1)
                                    elif sample_type == 1:
                                        digitid = gen_with_rule(type=0)
                                        # digitid = random.sample(list(PREFIX_SIGN), 1)[0] + digitid
                                        gen_amounts.append(digitid)
                                        newline = line[:start_index] + re.sub(res[3],   digitid,  line[start_index:], count=1)
                                    else:
                                        digitid = gen_with_rule(type=0)
                                        gen_amounts.append(digitid)
                                        newline = line[:start_index] + re.sub(res[3], digitid, line[start_index:], count=1)
                                    gen_types.append(getType(newline))
                                    buffer.append(newline)

                            newline = clean_line3(newline)
                            item['text'] = newline
                            entities = [a+"-"+b for a,b in zip(gen_amounts, gen_types)]
                            item['entities'] = entities
                            contract_list.append(item)
                        else:
                            # if getType(line.strip()) == "RES":
                            #     gen_type = random.randint(0, 2)
                            #     mis_formald = gen_with_rule(type=gen_type)
                            #     newline = line[:start_index] + re.sub(res[3], digitid, line[start_index:], count=1)
                            #     item = {}
                            #     item['text'] = newline
                            #     entities = [mis_formald + "-" + 'RES']
                            #     item['entities'] = entities
                            #     contract_list.append(item)
                            buffer.append(line.strip())

                # contracts = r"\n".join(buffer)
                #
                # contracts = contracts.replace("[", "", 10 ** 10)
                # contracts = contracts.replace("]", "", 10 ** 10)
                # contracts = contracts.replace("/", "", 10 ** 10)
                #
                # contracts = contracts.replace("▁", "", 10 ** 10)
                # contracts = contracts.replace("_", "", 10 ** 10)
                # contracts = contracts.replace("＿", "", 10 ** 10)
                #
                # contracts = contracts.replace("{.underline}", "", 10 ** 10)
                # index_marks = []
                # for amount in gen_amounts:
                #     start = contracts.find(amount)
                #     end = start + len(amount) -1
                #     if start>0:
                #         index_marks.append((start,end))
                # if index_marks:
                #     indexs =";".join( [str(k)+','+str(v) for (k,v) in index_marks] )
                #     contract_list.append({"contract":contracts,"indexes": indexs})
                # df = df.append({"contract":contracts,"indexes": indexs}, ignore_index=True)
# df = pandas.DataFrame(contract_list)
# df.to_csv("../data/contract_amount_train_v7.csv", columns=["contract", "indexes"], index=False)
with open('../data/amount_train_v8.json','w',encoding='utf-8') as fw:
    json.dump(contract_list, fw, ensure_ascii=False,indent=4)
print('FIN')