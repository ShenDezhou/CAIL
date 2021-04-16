import re
import pandas
from classmerge import class_dic, indic

def cleanhtml(raw_html):
    cleanr = re.compile('<[^>]+>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def cleanentity(raw_html):
    cleanr = re.compile('&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def cleanspaceholder(raw_html):
    cleanr = re.compile(r'[�╳x\-­_＿—－…×*├│┤｜ 　	┢┳┿┪┠─┼┨┏┯━┓┌┬┐┃┗┷┛└┴┘□]*')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def cleanduplicationfullcharacterchinese2(raw_html):
    cleanr = re.compile(r'[\uff01-\uff5e]*')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

# def shortenspace(raw_html):
#     cleanr = re.compile(r'[ ]{1,}')
#     cleantext = re.sub(cleanr, '', raw_html)
#     return cleantext
#
# def shortenspacelines(raw_html):
#     cleanr = re.compile(r' \n', re.M)
#     cleantext = re.sub(cleanr, '\n', raw_html)
#     return cleantext
#
# def shortenlinesspace(raw_html):
#     cleanr = re.compile(r'\n ', re.M)
#     cleantext = re.sub(cleanr, '\n', raw_html)
#     return cleantext

def shortenlines(raw_html):
    cleantext = str(raw_html).replace(r"\n", "", 10 ** 10)
    cleantext = cleantext.replace(r"\r","", 10**10)
    cleantext = cleantext.replace(r"?", "", 10 ** 10)
    cleantext = cleantext.replace(r" ", "", 10 ** 10)
    return cleantext


def cleanall(raw_html):
    #raw_html = shortenlines(raw_html)
    raw_html = cleanhtml(raw_html)
    raw_html = cleanentity(raw_html)
    raw_html = cleanspaceholder(raw_html)
    raw_html = cleanduplicationfullcharacterchinese2(raw_html)
    clean_html = shortenlines(raw_html)
    return clean_html

rawfile = "data/hetong_20200721.csv"

def clean():
    df = pandas.read_csv(rawfile, delimiter=',')
    df['title'] = df['title'].apply(shortenlines)
    df['content'] = df['content'].apply(cleanhtml)
    df['content'] = df['content'].apply(cleanentity)
    df['content'] = df['content'].apply(cleanspaceholder)
    df['content'] = df['content'].apply(cleanduplicationfullcharacterchinese2)
    df['content'] = df['content'].apply(shortenlines)


    train = df.loc[df['type1'] != '/']
    test = df.loc[df['type1'] == '/']
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)

#clean()



def cleanfile(filename=rawfile):
    df = pandas.read_csv(filename, delimiter=',')
    df['title'] = df['title'].apply(shortenlines)
    df['content'] = df['content'].apply(cleanhtml)
    df['content'] = df['content'].apply(cleanentity)
    df['content'] = df['content'].apply(cleanspaceholder)
    df['content'] = df['content'].apply(cleanduplicationfullcharacterchinese2)
    df['content'] = df['content'].apply(shortenlines)
    df.to_csv(filename, index=False)

#cleanfile("dataset/train-phase2.csv")

def enpercent(raw_html):
    cleanr = re.compile('[a-zA-Z]*')
    english_words = re.findall(cleanr, str(raw_html))
    counter=0
    for line in english_words:
        counter+=len(line)
    threshold = 0.9
    return (1.0*counter/len(str(raw_html))) < threshold


def sample():
    df = pandas.read_csv("data/train.csv")
    category = set(df['type1'])
    print(len(category), category)
    # sample
    df['enpercent'] = df['content'].map(enpercent)
    # df['content'] = df['content'].apply(cleanduplicationfullcharacterchinese2)

    dfsample = []
    for type in category:
        dfx = df.loc[df['type1'] == type]
        dfx = dfx.loc[dfx['enpercent']] # remove English articles.
        dfy = dfx.loc[dfx['content'].map(len) >= 50]
        dfz = dfy.sample(n=100, replace=True).copy()
        dfsample.append(dfz)

    print(len(dfsample))
    finaldf = pandas.concat(dfsample)
    finaldf.to_csv("data/dev.csv", columns=['type1','title','content'], index=False)

# sample()

def cleanupdev():
    df = pandas.read_csv("data/dev.csv")
    df['title'] = df['title'].apply(shortenlines)
    df['content'] = df['content'].apply(shortenlines)
    df['content'] = df['content'].apply(cleanduplicationfullcharacterchinese2)
    df.to_csv("data/dev_new.csv", columns=['type1','title','content'], index=False)

# cleanupdev()
def cleanminor():
    df = pandas.read_csv("data/train.csv", delimiter=',')
    df = df[(df['type1'] != '居间合同') & (df['type1'] != '托管合同') & (df['type1'] != '仓储合同') & (df['type1'] != '储运合同') & (df['type1'] != '供用合同')]
    df['typeindex'] = df['type1'].map(indic)
    groupby_count1 = df.groupby(['typeindex']).count()
    print(groupby_count1)
    df.to_csv("dataset/train.csv", columns=['type1','title','content'], index=False)

#cleanminor()