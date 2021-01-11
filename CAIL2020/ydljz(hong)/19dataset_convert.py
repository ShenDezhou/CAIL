import json
import re


results = []
_id = 0

def process_context(line):
    spans = re.split('([,，:;；。])', line)
    spans = [span for span in spans if len(span)>0]
    spans_sep = []
    for i in range(len(spans)//2):
        spans_sep.append(spans[2*i]+spans[2*i+1])
    if len(spans_sep) == 0:
        return []
    return [[spans_sep[0],spans_sep]]

def supporting_facts(answers, context_lines):
    res = []
    idx = set()
    answers = list(set(answers))
    for i in range(len(context_lines)):
        for answer in answers:
            if context_lines[i].find(answer) != -1:
                if i not in idx:
                    res.append([context_lines[0], i])
                idx.add(i)
    return res


with open('data/train19_train.json', 'w', encoding='utf8') as fw:

    fin = open('big_train_data.json', 'r', encoding='utf8')
    line = fin.readline()
    dic = json.loads(line)
    for item in dic['data']:
        id = item['caseid']
        domain = item['domain']
        para = item['paragraphs'][0]
        context = para['context']
        casename = para['casename']
        qas = para['qas']
        for qa in qas:
            question = qa['question']
            qid = qa['id']
            is_unknown = qa['is_impossible']
            answers = qa['answers']

            ans_starts = [ans['answer_start'] for ans in answers]
            if len(ans_starts) > 0:
                answer_pos = min(ans_starts)
            else:
                answer_pos = 0
            answer_start = max(answer_pos-150, 0)
            answer_end = min(answer_start+512, len(context))

            conv_dic = {}
            conv_dic['_id'] = _id
            conv_dic['context'] = process_context(context[answer_start:answer_end])
            conv_dic['question'] = question
            conv_dic['supporting_facts'] = []
            if is_unknown == "true":
                conv_dic['answer'] = "unknown"
            else:
                conv_dic['answer'] = answers[0]['text']
                ans_spans = [answer['text'] for answer in answers]
                conv_dic['supporting_facts'] = supporting_facts(ans_spans, conv_dic['context'][0][1])
            results.append(conv_dic)
            _id+= 1
    fin.close()
    fw.write(json.dumps(results, ensure_ascii=False, indent=4))
print('FIN')