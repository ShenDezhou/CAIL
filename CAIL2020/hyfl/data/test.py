import re

import pandas

pattern = re.compile('[A-Z]')

def match(line):
    if pattern.match(line):
        return True
    return False
df = pandas.read_csv('category.csv')
df['supreme'] = df['id'].map(match)
df = df[df['supreme']==True]
df.to_csv("supreme.csv",index=False)
print('FIN')
