import pandas
import lawrouge
from operator import itemgetter

'''
    Retrieve contract title to match the contract law articles from the 《CIVIL CODE 2020》.
'''



class TypeArticle():

    def __init__(self):
        self.contract_type = []
        with open('data/type-8.dic','r',encoding='utf-8') as f:
            for line in f:
                self.contract_type.append(line.strip())

        self.general_type = []
        with open('data/generaltype-8.dic', 'r', encoding='utf-8') as f:
            for line in f:
                self.general_type.append(line.strip())

        self.type_range_df = pandas.read_csv('data/typearticle.csv')
        self.general_type_range_df = pandas.read_csv('data/generalarticle.csv')
        self.code_contract_df = pandas.read_csv('dataraw/civil_code_contract.csv')
        self.rouge = lawrouge.Rouge()

    def get_weighted_score(self, contract_title, civilcode):
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
        typeindex, typedesc = self.get_weighted_score(contract_title, self.contract_type)
        type_range = self.type_range_df[self.type_range_df['type'] == typeindex + 1]

        for index, row in type_range.iterrows():
            start, to = row[1], row[2]
            print(start, to)
            break

        contract_df = self.code_contract_df[(self.code_contract_df['id'] <= to) & (self.code_contract_df['id'] >= start)]
        articles = contract_df['desc'].to_list()
        print(articles)
        return articles

    def findArticles_default_2(self, contract_title):
        general_typeindex, general_typedesc = self.get_weighted_score(contract_title, self.general_type)

        general_type_range = self.general_type_range_df[self.general_type_range_df['type'] == general_typeindex]

        for index, row in general_type_range.iterrows():
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

    contract_article = '3、违反本合同第四条第4款，延期交付的违约方应按延期交付天数每天人民币_________元向乙方支付违约金，并承担延误期内的风险责任。'
    for i in range(len(gridtype)):
        articles = typearticle.findArticles(gridtype[i])
        print(articles)
        index, article = typearticle.get_weighted_score(contract_article,articles)
        print(index, article)

        articles_default = typearticle.findArticles_default_2(contract_article)
        print(articles_default)
        index, article = typearticle.get_weighted_score(contract_article, articles_default)
        print(index, article)
        break

    print('FIN')
