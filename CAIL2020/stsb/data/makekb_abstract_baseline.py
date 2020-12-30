import json
import pandas


types = ['Education', 'Awards', 'Organization|Brand', 'Food', 'Software', 'Biological', 'Game', 'Software|Game', 'Medicine', 'Person|Other', 'Law&Regulation', 'Natural&Geography', 'Constellation', 'Location|Other', 'Person', 'Time&Calendar', 'Event|Work', 'Event', 'Location|Organization', 'Location', 'Work', 'Other', 'Culture', 'Brand', 'Brand|Organization', 'Disease&Symptom', 'Person|VirtualThings', 'Website', 'Vehicle', 'VirtualThings', 'Diagnosis&Treatment', 'Organization']
cn_types = ['教育','奖项','组织|品牌','食品','软件','生物','游戏','软件|游戏','医学','人|其他','法律法规' ,'自然地理','星座','位置|其他','人员','时间和日历','事件|工作','事件','位置|组织','位置','工作','其他' ,'文化','品牌','品牌|组织','疾病与症状','人|虚拟事物','网站','车辆',' 虚拟事物','诊断与治疗','组织']
type_dic = dict(zip(types,cn_types))

count = 0
collect = []
with open('kb.json','r',encoding='utf-8') as f:
    for line in f:
        item = json.loads(line.strip())
        word_id = item['subject_id']
        word = item['subject']
        data = item['data']
        en_type = item['type']
        cn_type = type_dic[en_type]

        abstract = [predictor['object'] for predictor in data if
                    predictor['predicate'] in ['摘要', '义项描述'] and predictor['predicate'] != "。"]

        core_abstract = ""
        if len(abstract):
            for ab in abstract:
                core_abstract += ab

        alias = item['alias']
        collect.append({'id':word_id,'word':word,'type':en_type,'cntype':cn_type, 'abstract': core_abstract})

        for alias_ in alias:
            collect.append({'id': word_id, 'word': alias_,'type':en_type,'cntype':cn_type, 'abstract': core_abstract})
        count += 1
        if count % 100 == 0:
            print('.')

df = pandas.DataFrame(collect)
df.to_csv('kb.csv', index=False)
print('FIN')