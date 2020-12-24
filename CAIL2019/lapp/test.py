import pandas
from gensim.models import KeyedVectors
from tqdm import tqdm

model = KeyedVectors.load_word2vec_format("model/doc-512.txt", binary=False, encoding='utf-8')

acc = 0
total = 0
for phase in ['train','valid','test']:
    df = pandas.read_csv('dataset/'+phase+'_id.csv')
    tobj = tqdm(df.itertuples(index=False), ncols=80)
    for row in tobj:
        aid, bid,cid = row
        aid, bid, cid = str(aid), str(bid), str(cid)
        close_score = model.similarity(aid, bid)
        dist_score = model.similarity(aid, cid)
        if close_score > dist_score:
            acc += 1
        total += 1
        tobj.set_description(desc=phase+",ACC="+str(acc*1.0/total))
    print(phase, "ACC="+str(acc*1.0/total))
print("FIN")

