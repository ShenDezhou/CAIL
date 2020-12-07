import os
import psutil
import re

#training score : 0.88
#租赁合同、借款合同、劳动合同、继承、侵权责任、追偿权
class CheckReason():
    def __init__(self):
        with open('reason.dic','r', encoding='utf-8') as f:
            words = f.readlines()
            words = [word.strip() for word in words]
            self.arguereasondic = dict(zip(words, range(len(words))))
        #self.arguereasondic = #dict(zip(['劳动合同', '侵权责任', '租赁合同', '借款合同', '继承', '追偿权', '借款', '侵权', '继承关系'], range(9)))
        pattern=""
        for word in words:
            pattern += "(%s)|" % word
        pattern= pattern.strip('|')
         # pattern = '(.{4})纠纷'
        self.pr = re.compile(pattern)

    def print_mem(self):
        process = psutil.Process(os.getpid())  # os.getpid()
        memInfo = process.memory_info()
        return '{:.4f}G'.format(1.0 * memInfo.rss / 1024 /1024 /1024)


    def checkArgueReason(self, text):
        new_reason = ""
        for dic in text:
            if self.pr.search(dic['sentence']):
                new_reason = self.pr.findall(dic['sentence'])[0]
                new_reason = [i for i in new_reason if len(i) > 0]
                if len(new_reason):
                    return "原被告系%s纠纷。" % new_reason[0]
                # for verify_r in self.arguereasondic.keys():
                #     if verify_r in new_reason:
                #         new_reason = verify_r
                #         return "原被告系%s纠纷。" % new_reason
                #add new reason to dicts.
                # self.arguereasondic[new_reason] = len(self.arguereasondic)
                # break
        return "原被告系纠纷关系。"

    def isArgueReason(self, sentence):
        if self.pr.search(sentence):
            new_reason = self.pr.findall(sentence)[0]
            new_reason = [i for i in new_reason if len(i) > 0]
            if len(new_reason):
                return new_reason[0]
            # for verify_r in self.arguereasondic.keys():
            #     if verify_r in new_reason:
            #         return verify_r
        return None