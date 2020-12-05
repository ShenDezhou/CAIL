
from gbtimportance.CheckImportance import CheckImportance
from gbtrank.CompRank import CompRank

import json

from rulepropose.CompPropose import CompPropose
from ruleopinion.CompOpinion import CompOpinion
from ruleresult.CompResult import CompResult
from rulereason.CheckReason import CheckReason
from rulefact.CompFact import CompFact

from filter.CheckWordImportance import CheckWordImportance
import fire

input_path = "/data/data.json"
output_path = "/output/result.json"

def main(in_file='/data/SMP-CAIL2020-test1.csv',
         out_file='/output/result1.csv'):
    importance = CheckImportance(cwd='gbtimportance')
    rank = CompRank(cwd='gbtrank')
    courtresult = CompResult()
    reason = CheckReason()
    propose = CompPropose()
    fact = CompFact()
    opinion = CompOpinion()
    filter = CheckWordImportance()
    with open(out_file, 'w', encoding='utf8') as fw:
        with open(in_file, 'r', encoding="utf8") as f:
            for line in f:
                data = json.loads(line)
                which_reason = ""

                id = data.get('id')
                text = data.get('text')  # "text": [{"sentence":"001"},{"sentence":"002"}]
                collected = []

                #2.原告起诉
                proposed = propose.getPropose(text)

                #3.查明
                facts = fact.getFact(text)

                #4.法院意见
                opinions = opinion.getOpinion(text)

                # 5.6 依据+判决结果
                trialresults = courtresult.getCourtResult(text)

                #1. 纠纷原因
                for i, sentence in enumerate(text):
                    if reason.isArgueReason(sentence["sentence"]):
                        which_reason = reason.isArgueReason(sentence["sentence"])
                        break

                data['reason']=which_reason
                dup = set()
                for ll in (proposed,facts,opinions,trialresults):
                    for line in ll:
                        if line not in dup:
                            collected.append({'sentence': line})
                            dup.add(line)

                data['text'] = collected
                #####OUTPUT#######
                #head
                # summary = []
                # # 1. 纠纷理由
                # if which_reason:
                #     summary.append("原被告系%s纠纷关系。" % which_reason)
                # else:
                #     summary.append(reason.checkArgueReason(text))
                # print(summary)
                #
                # #body
                # # 2.原告诉称; 3.查明 4.本院认为 5.6.法律依据、判决结果
                # def filtertool(somelist):
                #     newlist = [line for line in somelist if importance.checkImportance(line)]
                #     if len(newlist)==0:
                #         return somelist
                #     return newlist
                #
                # summary.extend(filtertool(proposed))
                # summary.extend(filtertool(facts))
                # summary.extend(filtertool(opinions))
                # summary.extend(trialresults)
                #
                # if len(collected) < 1000:
                #     pass
                # else:
                #     # collected = [sent for sent in collected if len(sent) > 0]
                #
                #     sentrank = []
                #     for sentence in collected:
                #         sentrank.append(rank.checkRank(sentence))
                #     rank2sent = dict()
                #     for key, v in zip(sentrank, collected):
                #         try:
                #             rank2sent[key].append(v)
                #         except KeyError:
                #             rank2sent[key] = [v]
                #     for key in sorted(rank2sent.keys()):
                #         summary.extend(rank2sent[key])
                #
                # # 过滤
                # # summary = [line for line in summary if importance.checkImportance(line)]
                # # summary = filter.checkWordImportance(summary)
                # summary = "".join(summary)
                # result = dict(
                #     id=id,
                #     summary=summary
                # )
                fw.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    fire.Fire(main)
