from judger.judger import Judger

INPUT='input/small/'
OUTPUT = 'output'
jud = Judger('accu.txt', 'law.txt')
res = jud.test(INPUT, OUTPUT)
print(res)
scor = jud.get_score(res)
print(scor)
print('FIN')