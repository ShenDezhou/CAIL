import json
import pandas as pd

smallfile="data/inter-valid.json"
df = pd.DataFrame()
with open(smallfile, 'r', encoding='utf-8') as fin:
    for line in fin:
        print('.')
        sents = json.loads(line.strip())
        for s in sents['pos']:
            dic = {'type':1, 'content': s}
            df = df.append(dic, ignore_index=True)
        for s in sents['neg']:
            dic = {'type':0, 'content': s}
            df = df.append(dic, ignore_index=True)
    df.to_csv("data/sfzy_core_valid.csv", columns=['type','content'], index=False)