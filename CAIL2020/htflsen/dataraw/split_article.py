import re

import pandas

rule = '　+第([^条]{1,5})条　(.*)'
chapter = '第[一二三四五六七八九十]{1,3}分?[章编]'
pattern = re.compile(rule)
chapter_pattern = re.compile(chapter)

FORMAL_DIGIT="零一二三四五六七八九"
math_digit="0123456789"

def format2digit(word):
    trans = ""
    if word.startswith('十'):
        trans += '1'

    for c in word:
        if c in FORMAL_DIGIT:
            trans += math_digit[FORMAL_DIGIT.index(c)]
        if word.endswith(c):
            if c=="十":
                trans += '0'
            if c=="百":
                trans += '00'
    return trans

df = pandas.DataFrame()
context = ""
with open('civil_code_contract.txt','r', encoding='utf-8') as f:
    buffer = []
    digit = 0
    for line in f:
        match = re.search(pattern, line)
        if match:
            #output
            article_digit = format2digit(match.group(1))
            if digit:
                tup = (str(int(article_digit)-1), r"\n".join(buffer))
                buffer = []
                dic = dict(zip(('id', 'desc'), tup))
                df = df.append(dic, ignore_index=True)
            buffer.append(line.strip())
            digit += 1
        else:
            if chapter_pattern.search(line):
                context = line.strip().split('　')[-1]
            else:
                buffer.append(line.strip())
    #last
    if buffer:
        tup = (article_digit, r"\n".join(buffer))
        dic = dict(zip(('id', 'desc'), tup))
        df = df.append(dic, ignore_index=True)

df.to_csv('civil_code_contract.csv', columns=['id','desc'], index=False)
print('FIN')