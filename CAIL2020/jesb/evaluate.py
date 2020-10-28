import pandas
import json
import urllib3
from six import unichr


def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += unichr(inside_code)
    return rstring

def compare(s,t):
    c = 0
    for i in s:
        ii = strQ2B(i)
        if ii in t:
            c+= 1
        else:
            print(ii, t)

    return c*1.0/len(s)


http = urllib3.PoolManager()
url = "http://localhost:58084/z"
df = pandas.read_csv("data/contract_amount_test.csv", encoding='utf-8')
correct = 0
total = len(df)
for x, row in df.iterrows():
    contract = row[0]
    indexes = row[1].split(";")
    tags = []
    for t in indexes:
        s,e = t.split(',')[0:2]
        s, e = int(s), int(e)
        tags.append(contract[s:e])

    encoded_data = json.dumps({'1': contract.replace('"','\\"')}).encode('utf-8')
    result = http.request('POST', url, headers={"Content-Type":"application/json"},
                          body=encoded_data)
    result = json.loads(result.data)
    model_predict = result["answer"]

    correct += compare(tags,model_predict)

    print("correct", correct, x)
print("ACC", float(correct*1.0/total))

