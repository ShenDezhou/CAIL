import json

import pandas
import urllib3
from classmerge import match
from dataclean import cleanall

df = pandas.read_csv("dataset/valid-phase1.csv")
http = urllib3.PoolManager()
correct = 0
for index, row in df.iterrows():
    label = row[0]
    title = row[1].replace(".doc","").replace(".docx","")
    content = cleanall(row[2])
    url = "http://192.168.0.161:58080/z?1={}&2={}".format(title, content)
    print(url)
    if len(url) > 9999:
        url = url[:9999]
    result = http.request('GET', url)
    result = json.loads(result.data)
    print(label, result['answer'][0])
    df.at[index, 'type1'] = result['answer'][0]
    df.at[index, 'title'] = title
    df.at[index, 'content'] = content
    if match(result['answer'][0], label):
        correct +=1
df.to_csv("eval/test-bert.csv", index=False)
print('ACCURACY:{}%'.format(correct*100.0/len(df)))