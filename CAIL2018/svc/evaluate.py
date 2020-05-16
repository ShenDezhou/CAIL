from judger.judger import Judger

jud = Judger('accu.txt', 'law.txt')
res = jud.test('input','output')
scor = jud.get_score(res)
print(scor)
print('FIN')