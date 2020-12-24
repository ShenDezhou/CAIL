import pandas

df = pandas.read_csv("total_scm.csv")
docs = []
for k in df.columns:
    dl = df[k].tolist()
    docs.extend(dl)
df_new = pandas.DataFrame(docs, columns=["document"])

df_new.to_csv("total_documents.csv", index_label='id')
print('FIN')