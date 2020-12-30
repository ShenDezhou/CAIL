import json
import pandas


count = 0
collect = []
with open('kb.json','r',encoding='utf-8') as f:
    for line in f:
        item = json.loads(line.strip())
        word_id = item['subject_id']
        word = item['subject']
        data = item['data']
        abstract = [predictor['object'] for predictor in data if predictor['predicate'] in ['摘要','义项描述'] and predictor['predicate'] !="。"]
        if len(abstract):
            abstract = abstract[0]
            if abstract == '。':
                abstract = ""
        else:
            abstract = ""
        alias = item['alias']
        collect.append({'id':word_id,'word':word, 'abstract': abstract})

        for alias_ in alias:
            collect.append({'id': word_id, 'word': alias_, 'abstract': abstract})
        count +=1
        if count % 100 == 0:
            print('.')

df = pandas.DataFrame(collect)
df.to_csv('kb.csv', index=False)
print('FIN')