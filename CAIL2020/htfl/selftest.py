import json

import pandas
import urllib3

df = pandas.read_csv("data/dev.csv")
http = urllib3.PoolManager()
correct = 0
for index, row in df.iterrows():
    label = row[0]
    url = "http://192.168.0.161:58080/z?1={}&2={}".format(row[1], row[2])
    print(url)
    if len(url)> 9999:
        url = url[:9999]
    result = http.request('GET', url)
    result = json.loads(result.data)
    print(label, result['answer'][0])
    if result['answer'][0]==label:
        correct +=1

print('ACCURACY:{}%'.format(correct*100.0/len(df)))