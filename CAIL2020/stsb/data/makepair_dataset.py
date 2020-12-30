import json
import pandas

df = pandas.read_csv('kb.csv')
rec_list = []
count = 0
with open('dev.json', 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line.strip())
        text = item['text']
        words = item['mention_data']

        new_words = []
        for word in words:
            id = word['kb_id']
            mention = word['mention']
            offset = word['offset']
            search = df[df['word'] == mention]

            def isMatch(tid):
                if str(tid) == str(id):
                    return 1
                return 0

            search['score'] = search['id'].map(isMatch)
            neg = 0
            for row in search.itertuples(index=False):
                if row[5]:
                    rec_list.append({'text':text, 'mention':mention, 'offset':offset,
                                     'abstract':row[4],'score':row[5]})
                else:
                    if neg == 0:
                        rec_list.append({'text': text, 'mention': mention, 'offset': offset,
                                     'abstract': row[4], 'score': row[5]})
                        neg+= 1
        count += 1
        if count % 100 == 0:
            print('.')

df_result = pandas.DataFrame(rec_list)
df_result.to_csv('dev_pair.csv', index=False)
print('FIN')
