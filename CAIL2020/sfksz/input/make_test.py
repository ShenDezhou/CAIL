import json

with open('train_data.json','r',encoding='utf-8') as fr:
    with open('train_data_train.json', 'w', encoding='utf-8') as fw1:
        with open('train_data_test.json', 'w', encoding='utf-8') as fw2:
            for line in fr:
                item = json.loads(line.strip())
                id = (int)(item['id'].split('_')[-1])
                if id % 4 == 0:
                    fw2.write(line)
                else:
                    fw1.write(line)
print('FIN')
