import json

with open('train_data.json','w',encoding='utf-8') as fw:
    for file in ['0_train.json','1_train.json']:
        with open(file,'r',encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                if len(item['answer']) > 0:
                    fw.write(line)
                else:
                    print(line)
print('FIN')