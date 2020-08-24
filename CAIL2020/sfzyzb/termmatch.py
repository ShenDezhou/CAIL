import json
import re
from difflib import SequenceMatcher
from query_sim import Query_Similar
import pandas

smallfile = "data/sfzy_small.json"
bigfile = "data/sfzy_big.json"
interfile = "data/inter.json"

# threshold = 0.3

def process_context(line):
    spans = re.split('([,，:;；。])', line)
    spans = [span for span in spans if len(span)>0]
    spans_sep = []
    for i in range(len(spans)//2):
        spans_sep.append(spans[2*i]+spans[2*i+1])
    if len(spans_sep) == 0:
        return []
    return spans_sep


with open(interfile, 'w', encoding='utf-8') as fw:
    with open(smallfile,'r', encoding='utf-8') as fin:
        for line in fin:
            sents = json.loads(line.strip())
            pos = []
            neg = []
            summary = sents['summary']
            text = sents['text']
            sentences = [item['sentence'] for item in text]
            summary_spans = process_context(summary)
            query_sim = Query_Similar(sentences)
            matching_ids = [query_sim.find_similar(span) for span in summary_spans]
            pos = [sentences[i] for i in range(len(sentences)) if i in matching_ids]
            neg = [sentences[i] for i in range(len(sentences)) if i not in matching_ids]
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
            summary_spans = process_context(summary)
            query_sim = Query_Similar(sentences)
            matching_ids = [query_sim.find_similar(span) for span in summary_spans]
            pos = [sentences[i] for i in range(len(sentences)) if i in matching_ids]
            neg = [sentences[i] for i in range(len(sentences)) if i not in matching_ids]
            sents['pos'] = pos
            sents['neg'] = neg
            print('.')
            fw.write(json.dumps(sents, ensure_ascii=False)+"\n")


tag_sents = []
with open(interfile, 'r', encoding='utf-8') as fin:
    for line in fin:
        print('.')
        sents = json.loads(line.strip())
        for s in sents['pos']:
            tag_sents.append((1,s))
        for s in sents['neg']:
            tag_sents.append((0,s))
    df = pandas.DataFrame(tag_sents, columns=['type','content'])
    df.to_csv("data/type_content_train.csv", columns=['type','content'], index=False)


# df = pandas.DataFrame()
tag_sents = []
with open(interfile, 'r', encoding='utf-8') as fin:
    for line in fin:
        print('.')
        sents = json.loads(line.strip())
        tag_sents.append(("".join(sents['pos']), sents['summary']))
    df = pandas.DataFrame(tag_sents, columns=['core', 'summary'])
    df.to_csv("data/core_summary_train.csv", columns=['core','summary'], index=False)