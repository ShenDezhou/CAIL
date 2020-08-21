import pandas

#train:2650,1594 valid:2650,1594
df = pandas.read_csv('core_summary_train.csv',encoding='utf8')
df['len']=df['core'].map(str).map(len)
print(max(df['len']))
df['len']=df['summary'].map(str).map(len)
print(max(df['len']))