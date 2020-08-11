import json

import pandas
import urllib3
from classmerge import class_case
from dataclean import cleanall

df = pandas.read_csv("data/test.csv")
http = urllib3.PoolManager()
correct = 0
for index, row in df.iterrows():
    label = row[0]
    title = str(row[1])#.replace(".doc","").replace(".docx","")
    content = cleanall(str(row[2]))
    url = "http://192.168.0.161:58081/z?1={}&2={}".format(title, content)
    print(url)
    if len(url) > 9999:
        url = url[:9999]
    result = http.request('GET', url)
    result = json.loads(result.data)
    print(class_case[label-1], result['answer'][0])
    # df.at[index, 'category'] = result['answer'][0]
    # df.at[index, 'title'] = title
    # df.at[index, 'fulltext'] = content
    if class_case.index(result['answer'][0]) == label - 1:
        correct +=1
#df.to_csv("eval/test-bert.csv", index=False)
print('ACCURACY:{}%'.format(correct*100.0/len(df)))