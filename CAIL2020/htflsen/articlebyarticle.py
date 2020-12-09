import pandas
import lawrouge
from operator import itemgetter

'''
    Retrieve contract title to match the contract law articles from the 《CIVIL CODE 2020》.
'''



class TypeArticle():

    def __init__(self):
        self.contract_type = []
        with open('data/codetype-8.dic','r',encoding='utf-8') as f:
            for line in f:
                self.contract_type.append(line.strip())

        self.type_range_df = pandas.read_csv('data/typearticle.csv')
        self.code_contract_df = pandas.read_csv('dataraw/civil_code_contract.csv')
        self.rouge = lawrouge.Rouge()

    def get_weighted_score(self, contract_title, civilcode=None):
        if not civilcode:
            civilcode = self.contract_type
        civilscore = []
        for i in range(len(civilcode)):
            scores = self.rouge.get_scores([contract_title], [civilcode[i]], avg=True)
            weighted_f1 = 0.2 * scores['rouge-1']['f'] + 0.4 * scores['rouge-2']['f'] + 0.4 * scores['rouge-l']['f']
            civilscore.append(weighted_f1)
        index, element = max(enumerate(civilscore), key=itemgetter(1))
        print('source:', contract_title, ', target:', civilcode[index], ', weighted f1:', element)
        return index, civilcode[index]

    def findArticles_default(self, contract_title):
        type_range = self.type_range_df[self.type_range_df['type'] == 0]

        for index, row in type_range.iterrows():
            start, to = row[1], row[2]
            print(start, to)
            break

        contract_df = self.code_contract_df[(self.code_contract_df['id'] <= to) & (self.code_contract_df['id'] >= start)]
        articles = contract_df['desc'].to_list()
        print(articles)
        return articles

    def findArticles(self, contract_title):
        typeindex, typedesc = self.get_weighted_score(contract_title)
        type_range = self.type_range_df[self.type_range_df['type'] == typeindex + 1]

        for index, row in type_range.iterrows():
            start, to = row[1], row[2]
            print(start, to)
            break

        contract_df = self.code_contract_df[(self.code_contract_df['id'] <= to) & (self.code_contract_df['id'] >= start)]
        articles = contract_df['desc'].to_list()
        print(articles)
        return articles

if __name__ == "__main__":
    typearticle = TypeArticle()
    # articles = typearticle.findArticles('买卖购销类合同')
    # print(articles)
    # print('FIN')


    gridtype = []
    with open('data/gridtype_firstclass.dic','r', encoding='utf-8') as f:
        for line in f:
            gridtype.append(line.strip(" │  ├─└12345678.\n"))

    print(gridtype)

    contract_article = '1、甲方应承诺出卖车辆不存在任何权属上的法律问题和各类尚未处理完毕的道路交通安全违法行为或者交通事故；应提供车辆的使用、维修、事故、检验以及是否办理抵押登记、交纳税费、报废期限等真实情况和信息。'
    for i in range(len(gridtype)):
        articles = typearticle.findArticles(gridtype[i])
        articles_default = typearticle.findArticles_default(gridtype[i])
        # print(articles)
        index, article = typearticle.get_weighted_score(contract_article,articles)
        print(index, article)
        index, article = typearticle.get_weighted_score(contract_article, articles_default)
        print(index, article)

    print('FIN')
