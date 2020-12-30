import json
import lawrouge
import pandas

df = pandas.read_csv('data/kb_full.csv')

typd_df = df['type'].sample(n=32)
types = ['Education', 'Awards', 'Organization|Brand', 'Food', 'Software', 'Biological', 'Game', 'Software|Game', 'Medicine', 'Person|Other', 'Law&Regulation', 'Natural&Geography', 'Constellation', 'Location|Other', 'Person', 'Time&Calendar', 'Event|Work', 'Event', 'Location|Organization', 'Location', 'Work', 'Other', 'Culture', 'Brand', 'Brand|Organization', 'Disease&Symptom', 'Person|VirtualThings', 'Website', 'Vehicle', 'VirtualThings', 'Diagnosis&Treatment', 'Organization']
cn_types = ['教育','奖项','组织|品牌','食品','软件','生物','游戏','软件|游戏','医学','人|其他','法律法规' ,'自然地理','星座','位置|其他','人员','时间和日历','事件|工作','事件','位置|组织','位置','工作','其他' ,'文化','品牌','品牌|组织','疾病与症状','人|虚拟事物','网站','车辆',' 虚拟事物','诊断与治疗','组织']
type_dic = dict(zip(cn_types, types))


lawrouge = lawrouge.Rouge(exclusive=True)

with open('model/full/result.json','w', encoding='utf-8') as fw:
    with open('data/dev.json','r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            text = item['text']
            words = item['mention_data']


            def get_score(target):
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
                    typd_df['type_score'] = typd_df['abstract'].map(get_score)
                    maxtype = typd_df.loc[typd_df['type_score'].idxmax()]
                    entype = type_dic[maxtype['type']]
                    id = "NIL_" + entype
                    print(mention,id, maxtype['type_score'])
                new_words.append({"kb_id": id,"mention": mention,"offset":offset})
            item['mention_data'] = new_words
            fw.write(json.dumps(item, ensure_ascii=False)+"\n")
print('FIN')
