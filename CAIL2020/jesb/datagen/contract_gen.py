import os
import random

import pandas

dic=[]
with open('amount.dic','r',encoding='utf-8') as f:
    lines = f.readlines()
    dic.extend([l.strip() for l in lines])

print(len(dic))
def trigger(line):
    for i in range(len(dic)):
        if dic[i] in line:
            return i, dic[i]
    return None

FORMAL_DIGIT="零一二三四五六七八九十百千万亿兆〇零壹贰叁肆伍陆柒捌玖拾佰仟萬億兆"
math_digit="1234567890,\uFF0C\uFF10\uFF11\uFF12\uFF13\uFF14\uFF15\uFF16\uFF17\uFF18\uFF19"
amount_len = (5, 15)
category_size = 100

def gen_amount(formal = True):
    len = random.randint(*amount_len)
    if formal:
        return "".join(random.sample(list(FORMAL_DIGIT), len))
    return "".join(random.sample(list(math_digit), len))


df = pandas.DataFrame(columns=["contract","indexes"])

for dirpath, dnames, fnames in os.walk("txt/"):
    for file in fnames:
        filename = os.path.join(dirpath, file)
        for _ in range(category_size):
            with open(filename, 'r', encoding='utf-8') as fr:
                buffer = []
                gen_amounts = []
                for line in fr:
                    res = trigger(line)
                    if res:
                        formald = gen_amount()
                        mathd = gen_amount(False)
                        gen_amounts.append(formald)
                        gen_amounts.append(mathd)
                        newline = line.replace(res[1], res[1] + formald + ("".join(random.sample(line, random.randint(1, len(line)))).replace("\n", "", 10 ** 10)) +mathd)
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
                    index_marks.append((start,end))
                if index_marks:
                    indexs =";".join( [str(k)+','+str(v) for (k,v) in index_marks] )
                    df = df.append({"contract":contracts,"indexes": indexs}, ignore_index=True)
df.to_csv("contract_amount_train.csv", columns=["contract", "indexes"], index=False)
print('FIN')