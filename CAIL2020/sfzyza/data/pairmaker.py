import json
import pandas as pd

smallfile="data/inter.json"
df = pd.DataFrame()
with open(smallfile, 'r', encoding='utf-8') as fin:
    for line in fin:
        print('.')
        sents = json.loads(line.strip())
        core_sents = "。".join(sents['pos'])
        dic = {'core':core_sents,"summary":sents['summary']}
        df = df.append(dic, ignore_index=True)
    print(df.shape)
    df.to_csv("data/core_summary_train.csv", columns=['core','summary'], index=False)