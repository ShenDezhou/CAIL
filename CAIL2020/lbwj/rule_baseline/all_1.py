import pandas as pd
import jieba
import numpy as np
import sys

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise Exception("run this script: "
                        "python eval.py $golds_file_path $predict_file_path")
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    df1 = pd.read_csv(input_file)
    df2 = pd.DataFrame(columns=['id','answer'])

    for i in range(len(df1)):
        df2.loc[i,'id'] = df1.loc[i,'id']
        df2.loc[i,'answer'] = 1

    df2['id'] = df2['id'].astype('int')
    df2['answer'] = df2['answer'].astype('int')
    df2.to_csv(output_file,encoding='utf-8',index=False)