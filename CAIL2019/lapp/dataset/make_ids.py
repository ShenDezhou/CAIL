import pandas

dic = pandas.read_csv('total_documents.csv')
doc_dic = dict(zip(dic['document'].tolist(), dic['id'].tolist()))

for phase in ['train','valid','test']:
    pair_close = pandas.read_csv("../data/"+phase+'.csv')

    tup = []

    for row in pair_close.itertuples(index=False):
        a,b,c = row
        aid, bid, cid = doc_dic[a], doc_dic[b], doc_dic[c]
        tup.append((aid, bid, cid))

    id_df = pandas.DataFrame(tup)
    id_df.to_csv(phase+"_id.csv", index=False)
print('FIN')

