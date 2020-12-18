import re

import pandas

rule = '^[A-Z0-9]+$'
pattern = re.compile(rule)

df = pandas.DataFrame()
context = ""
last_article_digit = ""
with open('filter.txt','r', encoding='utf-8') as f:
    buffer = []
    digit = 0
    for line in f:
        match = re.search(pattern, line)
        if match:
            #output
            article_digit = match.group(0)
            if digit:
                tup = (last_article_digit, r"\n".join(buffer))
                buffer = []
                dic = dict(zip(('id', 'desc'), tup))
                df = df.append(dic, ignore_index=True)
            # buffer.append(line.strip())
            last_article_digit = article_digit
            digit += 1
        else:
            buffer.append(line.strip())
    #last
    if buffer:
        tup = (article_digit, r"\n".join(buffer))
        dic = dict(zip(('id', 'desc'), tup))
        df = df.append(dic, ignore_index=True)

df['id'] = df['id'].astype(str)
df.to_csv("category.csv", columns=['id','desc'], index=False)

print('FIN')
