import json

import pandas


for phase in ['train','valid','test']:
    df = pandas.DataFrame()
    with open(phase+'.json','r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            if item['label'] == "B":
                sort_item = item
            else:
                sort_item = {"A":item["A"], "B":item["C"], "C":item["B"]}
            df = df.append(sort_item, ignore_index=True)
    df.to_csv(phase+'.csv', columns=['A','B','C'], index=False)
    print('FIN')