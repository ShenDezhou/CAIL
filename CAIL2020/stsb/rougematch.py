import json
import lawrouge
import pandas

df = pandas.read_csv('data/kb.csv')

lawrouge = lawrouge.Rouge(exclusive=True)

with open('data/result.json','w', encoding='utf-8') as fw:
    with open('data/dev.json','r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            text = item['text']
            words = item['mention_data']


            def get_score(target):
                if target =='。':
                    return 0
                if len(text)>0 and len(target)>0:
                    return lawrouge.get_scores(text,target, avg=2)['f']
                return 0

            new_words = []
            for word in words:
                mention = word['mention']
                offset = word['offset']
                search = df[df['word'] == mention]
                if len(search):
                    search['score'] = search['abstract'].map(get_score)
                    maxmention = search.loc[search['score'].idxmax()]
                    id = int(maxmention['id'])
                    score = maxmention['score']
                    print(mention,id, score)
                else:
                    id = "NIL_Work"
                    print(mention,-1)
                new_words.append({"kb_id": id,"mention": mention,"offset":offset})
            item['mention_data'] = new_words
            fw.write(json.dumps(item, ensure_ascii=False)+"\n")
print('FIN')
