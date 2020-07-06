import thulac
import re


class summary():
    def __init__(self):
        self.cut = thulac.thulac(seg_only=False)
        # n/名词 np/人名 ns/地名 ni/机构名 nz/其它专名   u/助词,  m/数词 q/量词 mq/数量词 t/时间词 f/方位词 s/处所词
        self.filter = ['np', 'ns', 'ni', 'nz', 'u','m','q', 'mq', 't','f', 's']
        self.mean_filter = ['n', 'np', 'ns', 'ni', 'nz', 'u', 'm', 'q', 'mq', 't', 'f', 's']

    def cut_text(self, alltext):
        count = 0
        train_text = []
        for text in alltext:
            count += 1
            if count % 2000 == 0:
                print(count)
            train_text.append(self.cut.cut(text, text=True))
        return train_text

    # def print_mem(self):
    #     process = psutil.Process(os.getpid())  # os.getpid()
    #     memInfo = process.memory_info()
    #     return '{:.4f}G'.format(1.0 * memInfo.rss / 1024 /1024 /1024)

    def summarize(self, statement):
        train_data = self.cut_text(statement)
        filterd = []
        for line in train_data:
            filterline = []
            for seg in line.split(" "):
                parts = seg.split("_")
                if len(parts) < 2:
                    continue
                if '中华人民共和国' in parts[0]:
                    pass
                elif parts[1] in self.filter:
                    continue
                filterline.append(parts[0])
            nerfilered = "".join(filterline)
            nerfilered = re.sub(u"\\(.*?\\)|（.*?）|\\{.*?}|\\[.*?]", "", nerfilered)
            filterd.append(nerfilered)
        return filterd

    def mean_summarize(self, statement):
        train_data = self.cut_text(statement)
        filterd = []
        for line in train_data:
            filterline = []
            for seg in line.split(" "):
                parts = seg.split("_")
                if len(parts) < 2:
                    continue
                if '中华人民共和国' in parts[0]:
                    pass
                elif parts[1] in self.mean_filter:
                    continue
                filterline.append(parts[0])
            nerfilered = "".join(filterline)
            nerfilered = re.sub(u"\\(.*?\\)|（.*?）|\\{.*?}|\\[.*?]", "", nerfilered)
            filterd.append(nerfilered)
        return filterd
