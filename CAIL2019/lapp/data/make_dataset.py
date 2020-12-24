import pandas

df = pandas.read_csv('train.csv')
df_valid = pandas.read_csv('valid.csv')
df_test = pandas.read_csv('test.csv')
df = df.append(df_valid,ignore_index=True)
df = df.append(df_test,ignore_index=True)
df.to_csv("total_scm.csv", index=False)
print('FIN')