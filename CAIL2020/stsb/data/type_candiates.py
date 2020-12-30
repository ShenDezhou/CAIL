import json
import pandas


types = ['Education', 'Awards', 'Organization|Brand', 'Food', 'Software', 'Biological', 'Game', 'Software|Game', 'Medicine', 'Person|Other', 'Law&Regulation', 'Natural&Geography', 'Constellation', 'Location|Other', 'Person', 'Time&Calendar', 'Event|Work', 'Event', 'Location|Organization', 'Location', 'Work', 'Other', 'Culture', 'Brand', 'Brand|Organization', 'Disease&Symptom', 'Person|VirtualThings', 'Website', 'Vehicle', 'VirtualThings', 'Diagnosis&Treatment', 'Organization']

typelist = set()
with open('kb.json','r',encoding='utf-8') as f:
    for line in f:
        item = json.loads(line.strip())
        typelist.add(item['type'])
print(typelist)
print('FIN')
