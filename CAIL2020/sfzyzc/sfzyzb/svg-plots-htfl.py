import matplotlib.pyplot as plt
import pandas


ls = []
fs = ['train_acc','train_f1','valid_acc','valid_f1']
df = pandas.read_csv('BERT-htfl-epoch.csv', dtype=dict(zip(fs,[float]*len(fs))))

def draw(fs):
    plt.style.use(['science', 'no-latex', 'grid'])
    plt.figure()
    for k in fs:
        ls.append(plt.plot(df['epoch'],df[k]))
    plt.xlabel('epoch')
    plt.ylabel('score')

    plt.legend(labels=fs,loc='best')
    plt.savefig('htfl-test.svg')
    # plt.show()

draw(fs)
