import json
from difflib import SequenceMatcher

import pandas

smallfile = "data/sfzy_small.json"
bigfile = "data/sfzy_big.json"
interfile = "data/inter.json"

threshold = 0.3
with open(interfile, 'w', encoding='utf-8') as fw:
    with open(smallfile,'r', encoding='utf-8') as fin:
        for line in fin:
            sents = json.loads(line.strip())
            pos = []
            neg = []
            summary = sents['summary']
            text = sents['text']
            sentences = [item['sentence'] for item in text]
            summary_spans = summary.split('。')
            for span in summary_spans:
                for s in sentences:
                    meter = SequenceMatcher(None, s, span).ratio()
                    if meter >= threshold:
                        pos.append(s)
            for s in sentences:
                if s not in pos:
                    neg.append(s)
            sents['pos'] = pos
            sents['neg'] = neg
            print('.')
            fw.write(json.dumps(sents, ensure_ascii=False)+"\n")

    with open(bigfile,'r', encoding='utf-8') as fin:
        for line in fin:
            sents = json.loads(line.strip())
            pos = []
            neg = []
            summary = sents['summary']
            text = sents['text']
            sentences = [item['sentence'] for item in text]
            summary_spans = summary.split('。')
            for span in summary_spans:
                for s in sentences:
                    meter = SequenceMatcher(None, s, span).ratio()
                    if meter >= threshold:
                        pos.append(s)
            for s in sentences:
                if s not in pos:
                    neg.append(s)
            sents['pos'] = pos
            sents['neg'] = neg
            print('.')
            fw.write(json.dumps(sents, ensure_ascii=False)+"\n")


df = pandas.DataFrame()
with open(interfile, 'r', encoding='utf-8') as fin:
    for line in fin:
        print('.')
        sents = json.loads(line.strip())
        for s in sents['pos']:
            dic = {'type':1, 'content': s}
            df = df.append(dic, ignore_index=True)
        for s in sents['neg']:
            dic = {'type':0, 'content': s}
            df = df.append(dic, ignore_index=True)
    df['type'] = df['type'].map(int)
    df.to_csv("data/type_content_train.csv", columns=['type','content'], index=False)


df = pandas.DataFrame()
with open(interfile, 'r', encoding='utf-8') as fin:
    for line in fin:
        print('.')
        sents = json.loads(line.strip())
        core_sents = "。".join(sents['pos'])
        dic = {'core':core_sents,"summary":sents['summary']}
        df = df.append(dic, ignore_index=True)
    df.to_csv("data/core_summary_train.csv", columns=['core','summary'], index=False)