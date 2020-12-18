import lawrouge
import pandas

rouge = lawrouge.Rouge(isChinese=True)

test = '宗教组织'
def get_score(candidates):
    line = test
    if candidates:
        candidates = str(candidates).split(r'\n')
        if len(candidates) > 1:
            scores = [rouge.get_scores([line], [candidate], avg=2) for candidate in candidates]
            scores = [score['f'] for score in scores]
            return max(scores)
        else:
            score = rouge.get_scores([line], [str(candidates)], avg=2)
            return score['f']
    return 0

df = pandas.read_csv('data/category.csv')
df['score'] = df['desc'].map(get_score)
index = df['score'].argmax()
print(df.iloc[index].to_dict())
print('FIN')