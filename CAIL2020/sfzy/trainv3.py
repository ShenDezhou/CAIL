
from gbtimportance.CheckImportance import CheckImportance
from gbtrank.CompRank import CompRank

import json
import re

from rulepropose.CompPropose import CompPropose
from ruleopinion.CompOpinion import CompOpinion
from ruleresult.CompResult import CompResult
from rulereason.CheckReason import CheckReason
from rulefact.CompFact import CompFact

from filter.CheckWordImportance import CheckWordImportance

input_path = "data/sfzy_small.json"
output_path = "output/result.json"

#
# def get_summary(text):
#     for i, _ in enumerate(text):
#         sent_text = text[i]["sentence"]
#         if re.search(r"诉讼请求：", sent_text):
#             text0 = text[i]["sentence"]
#             text1 = text[i + 1]["sentence"]
#             text2 = text[i + 2]["sentence"]
#             break
#         else:
#             text0 = text[11]["sentence"]
#             text1 = text[12]["sentence"]
#             text2 = text[13]["sentence"]
#     result = text0 + text1 + text2
#     return result


if __name__ == "__main__":
    importance = CheckImportance(cwd='gbtimportance')
    rank = CompRank(cwd='gbtrank')
    courtresult = CompResult()
    reason = CheckReason()
    propose = CompPropose()
    fact = CompFact()
    opinion = CompOpinion()
    filter = CheckWordImportance()
    with open(output_path, 'w', encoding='utf8') as fw:
        with open(input_path, 'r', encoding="utf8") as f:
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


                #####OUTPUT#######
                #head
                summary = []
                # 1. 纠纷理由
                if which_reason:
                    summary.append("原被告系%s纠纷关系。" % which_reason)
                else:
                    summary.append(reason.checkArgueReason(text))
                print(summary)

                #body
                # 2.原告诉称; 3.查明 4.本院认为 5.6.法律依据、判决结果
                def filtertool(somelist):
                    newlist = [line for line in somelist if importance.checkImportance(line)]
                    if len(newlist)==0:
                        return somelist
                    return newlist

                summary.extend(filtertool(proposed))
                summary.extend(filtertool(facts))
                summary.extend(filtertool(opinions))
                summary.extend(trialresults)

                if len(collected) < 1000:
                    pass
                else:
                    # collected = [sent for sent in collected if len(sent) > 0]

                    sentrank = []
                    for sentence in collected:
                        sentrank.append(rank.checkRank(sentence))
                    rank2sent = dict()
                    for key, v in zip(sentrank, collected):
                        try:
                            rank2sent[key].append(v)
                        except KeyError:
                            rank2sent[key] = [v]
                    for key in sorted(rank2sent.keys()):
                        summary.extend(rank2sent[key])

                # 过滤
                # summary = [line for line in summary if importance.checkImportance(line)]
                summary = filter.checkWordImportance(summary)
                summary = "".join(summary)
                result = dict(
                    id=id,
                    summary=summary
                )
                fw.write(json.dumps(result, ensure_ascii=False) + '\n')
