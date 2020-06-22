import json

input_gold = open('../data/sfzy_raw_input.json', mode='r', encoding='utf-8')
fo = open("system/gold_sum.1.txt", mode='w', encoding='utf-8')
for file in input_gold:
    readjson = json.loads(file)
    fo.write(readjson['summary']+'\n')

input_model = open('../output/result.json', mode='r', encoding='utf-8')
fo = open("model/model_sum.1.txt", mode='w', encoding='utf-8')
for file in input_model:
    readjson = json.loads(file)
    fo.write(readjson['summary']+'\n')

