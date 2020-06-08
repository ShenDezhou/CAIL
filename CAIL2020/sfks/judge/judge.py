import json
import os

#gold files
data_path="../input"
filename_list="0_train.json,1_train.json".replace(" ", "").split(",")
#model output file
model_output = "../output/result.txt"

# def precision():
#gold
res = {}

for filename in filename_list:
    f = open(os.path.join(data_path, filename), "r", encoding='utf-8')
    for line in f:
        x = json.loads(line)
        res[x["id"]] = "".join(x["answer"])

print(len(res))
#
modelres = json.load(open(model_output, "r", encoding='utf-8'))
print(len(modelres))

y=[]
yp=[]
for k in res.keys():
    g = res[k]
    g = [l for l in g if l != '。']
    if type(g) is str:
        y.append(g)
    if type(g) is list:
        if len(g) == 0:
            y.append('NNNN')
        else:
            y.append("".join(g))
    z = modelres[k][0]
    if type(z) is str:
        yp.append(z)
    else:
        yp.append("".join(z))
    #print(k, y[-1], yp[-1])

count=0
for i in range(len(y)):
    # y[i] = y[i]+'N'*(4-len(y[i]))
    # yp[i] = yp[i] + 'N' * (4 - len(yp[i]))
    if y[i] == yp[i]:
        count += 1

print("precision:", count*1.0/len(y))


