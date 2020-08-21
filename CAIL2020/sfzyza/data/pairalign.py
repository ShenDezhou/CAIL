import pandas as pd

df = pd.read_csv('data/core_summary_train.csv', encoding='utf8')
df_span = pd.DataFrame()

for row in df.itertuples(index=False):
    core, summary = str(row[0]), str(row[1])
    if len(core) < 512:
        dic = {'core':core, 'summary':summary}
        df_span = df_span.append(dic, ignore_index=True)
    else:
        core_lines = core.split('。')
        core_lines = [line for line in core_lines if len(line)> 0]
        summary_liens = summary.split("。")
        print(len(core_lines), len(summary_liens))
        if len(core_lines) == len(summary_liens):
            for ind in range(len(core_lines)):
                dic = {'core': core_lines[ind], 'summary': summary_liens[ind]}
                df_span = df_span.append(dic, ignore_index=True)
        elif len(core_lines) > len(summary_liens):
            diff = len(core_lines) - len(summary_liens)
            a = [0] * len(core_lines)
            span = len(core_lines) // len(summary_liens)
            for i in range(len(a)):
                if i!=0 and i % span == 0:
                    a[i] = span
            a[-1] = diff - sum(a)

            for ind in range(len(core_lines)):
                goback = sum(a[:ind])
                dic = {'core': core_lines[ind], 'summary': summary_liens[ind-goback]}
                print(dic)
                df_span = df_span.append(dic, ignore_index=True)
        else:
            assert False
            diff = len(core_lines) - len(summary_liens)
            a = [0] * len(core_lines)
            span = len(core_lines) // len(summary_liens)
            for i in range(len(a)):
                if i != 0 and i % span == 0:
                    a[i] = span
            a[-1] = diff - sum(a)

            for ind in range(len(core_lines)):
                goback = sum(a[:ind])
                dic = {'core': core_lines[ind], 'summary': summary_liens[ind - goback]}
                print(dic)
                df_span = df_span.append(dic, ignore_index=True)


def shortenlines(raw_html):
    cleantext = str(raw_html).replace(r"\n", "", 10 ** 10)
    cleantext = cleantext.replace(r"\r","", 10**10)
    cleantext = cleantext.replace(r"?", "", 10 ** 10)
    cleantext = cleantext.replace(r" ", "", 10 ** 10)
    return cleantext

df['core']=df['core'].map(shortenlines)
df['summary']=df['summary'].map(shortenlines)
df_span.to_csv('data/core_summary_span_train.csv', index=False)



