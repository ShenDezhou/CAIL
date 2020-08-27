import os
import psutil
import re

#training score :  0.738

class CompResult():
    def __init__(self, cwd=".", tfidf='statement_tfidf.model', gbt='statement_som_gbt.model'):
        patterns1 = "依照"
        patterns2 = "判决如下"
        self.ps1 = re.compile(patterns1)
        self.ps2 = re.compile(patterns2)
        patterne1 = '案件受理费'
        patterne2 = '如不服本判决'
        self.pe1 = re.compile(patterne1)
        self.pe2 = re.compile(patterne2)


    def cut_text(self, alltext):
        count = 0
        train_text = []
        for text in alltext:
            count += 1
            if count % 2000 == 0:
                print(count)
            train_text.append(self.cut.cut(text, text=True))

        return train_text

    def print_mem(self):
        process = psutil.Process(os.getpid())  # os.getpid()
        memInfo = process.memory_info()
        return '{:.4f}G'.format(1.0 * memInfo.rss / 1024 /1024 /1024)


    def checkResult(self, statement):
        train_data = self.cut_text([statement])
        vec = self.tfidf.transform(train_data)
        yp = self.gbt.predict(vec)
        return yp[0]

    def getCourtResult(self, sentences):

        tap = False
        cr = []
        for dic in sentences:
            if tap and (self.pe1.search(dic['sentence']) or self.pe2.search(dic['sentence'])):
                break

            if self.ps1.search(dic['sentence']) or self.ps2.search(dic['sentence']):
                tap = True

            if tap:
                cr.append(dic['sentence'])

        # print(cr)
        return cr
