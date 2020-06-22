import json

input_gold = open('../data/sfzy_raw_input.json', mode='r', encoding='utf-8')
count = 1

for file in input_gold:
    readjson = json.loads(file)
    fo = open("system/gold_sum.%d.txt" % count, mode='w', encoding='utf-8')
    fo.write(readjson['summary']+'\n')
    count += 1

input_model = open('../output/result.json', mode='r', encoding='utf-8')
count = 1

for file in input_model:
    readjson = json.loads(file)
    fo = open("model/model_sum.%d.txt" % count, mode='w', encoding='utf-8')
    fo.write(readjson['summary']+'\n')
    count += 1
