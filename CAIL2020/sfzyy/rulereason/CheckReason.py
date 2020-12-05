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
            self.arguereasondic = dict(zip([words, range(words)]))
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
                match = self.pr.findall(dic['sentence'])
                return "原被告系%s纠纷。" % match[0]
                # for verify_r in self.arguereasondic.keys():
                #     if verify_r in new_reason:
                #         new_reason = verify_r
                #         return "原被告系%s纠纷。" % new_reason
                #add new reason to dicts.
                # self.arguereasondic[new_reason] = len(self.arguereasondic)
                # break
        return "原被告系%s纠纷关系。" % new_reason

    def isArgueReason(self, sentence):
        if self.pr.search(sentence):
            new_reason = self.pr.findall(sentence)[0]
            return new_reason
            # for verify_r in self.arguereasondic.keys():
            #     if verify_r in new_reason:
            #         return verify_r
        return None