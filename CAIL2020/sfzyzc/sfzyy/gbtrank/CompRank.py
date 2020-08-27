import os
import thulac
import psutil
import joblib

#training score :  0.738

class CompRank():
    def __init__(self, cwd=".", tfidf='statement_tfidf.model', gbt='statement_som_gbt.model'):
        print('train tfidf...', self.print_mem())
        self.tfidf = joblib.load(os.path.join(cwd, tfidf))
        print('train gbt...', self.print_mem())
        self.gbt = joblib.load(os.path.join(cwd, gbt))
        self.cut = thulac.thulac(seg_only=True)

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


    def checkRank(self, statement):
        train_data = self.cut_text([statement])
        vec = self.tfidf.transform(train_data)
        yp = self.gbt.predict(vec)
        return yp[0]
