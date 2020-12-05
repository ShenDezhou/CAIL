
# from gbtimportance.CheckImportance import CheckImportance
# from gbtrank.CompRank import CompRank

import json

from rulepropose.CompPropose import CompPropose
from ruleopinion.CompOpinion import CompOpinion
from ruleresult.CompResult import CompResult
from rulereason.CheckReason import CheckReason
from rulefact.CompFact import CompFact

import fire

input_path = "/data/data.json"
output_path = "/output/result.json"

class Segment_Abstract(object):

    def __init__(self):
        self.courtresult = CompResult()
        self.reason = CheckReason()
        self.propose = CompPropose()
        self.fact = CompFact()
        self.opinion = CompOpinion()


    def get_abstract(self, data):
        """
        :intput: document sentence list
        :return: abstract sentence list
        """
        which_reason = ""

        id = data.get('id')
        text = data.get('text')  # "text": [{"sentence":"001"},{"sentence":"002"}]
        collected = []

        # 2.原告起诉
        proposed = self.propose.getPropose(text)

        # 3.查明
        facts = self.fact.getFact(text)

        # 4.法院意见
        opinions = self.opinion.getOpinion(text)

        # 5.6 依据+判决结果
        trialresults = self.courtresult.getCourtResult(text)

        # 1. 纠纷原因
        for i, sentence in enumerate(text):
            which_reason = self.reason.isArgueReason(sentence["sentence"])
            if which_reason:
                break

        data['reason'] = which_reason
        dup = set()
        for ll in (proposed, facts, opinions, trialresults):
            for line in ll:
                if line not in dup:
                    collected.append({'sentence': line})
                    dup.add(line)

        data['text'] = collected
        return data


def main(in_file='/data/SMP-CAIL2020-test1.csv',
         out_file='/output/result1.csv'):
    # importance = CheckImportance(cwd='gbtimportance')
    # rank = CompRank(cwd='gbtrank')
    courtresult = CompResult()
    reason = CheckReason()
    propose = CompPropose()
    fact = CompFact()
    opinion = CompOpinion()
    #filter = CheckWordImportance()
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
                    which_reason = reason.isArgueReason(sentence["sentence"])
                    if which_reason:
                        break

                data['reason']=which_reason
                dup = set()
                for ll in (proposed,facts,opinions,trialresults):
                    for line in ll:
                        if line not in dup:
                            collected.append({'sentence': line})
                            dup.add(line)

                data['text'] = collected
                fw.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    fire.Fire(main)
