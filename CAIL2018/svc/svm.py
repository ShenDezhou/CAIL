import os

from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
import json
from predictor import data
from sklearn.svm import LinearSVC
import joblib
import thulac
import psutil


dim = 5000


def cut_text(alltext):
	count = 0	
	cut = thulac.thulac(seg_only = True)
	train_text = []
	for text in alltext:
		count += 1
		if count % 2000 == 0:
			print(count)
		train_text.append(cut.cut(text, text = True))
	
	return train_text


def train_tfidf(train_data):
	tfidf = TFIDF(
			min_df = 5,
			max_features = dim,
			ngram_range = (1, 3),
			use_idf = 1,
			smooth_idf = 1
			)
	tfidf.fit(train_data)
	
	return tfidf



def read_trainData(path):
	fin = open(path, 'r', encoding = 'utf8')
	
	alltext = []
	
	accu_label = []
	law_label = []
	time_label = []

	line = fin.readline()
	while line:
		d = json.loads(line)
		alltext.append(d['fact'])
		accu_label.append(data.getlabel(d, 'accu'))
		law_label.append(data.getlabel(d, 'law'))
		time_label.append(data.getlabel(d, 'time'))
		line = fin.readline()
	fin.close()

	return alltext, accu_label, law_label, time_label


def train_SVC(vec, label):
	SVC = LinearSVC()
	SVC.fit(vec, label)
	return SVC

def print_mem():
	process = psutil.Process(os.getpid())  # os.getpid()
	memInfo = process.memory_info()
	return '{:.4f}G'.format(1.0 * memInfo.rss / 1024 /1024 /1024)

if __name__ == '__main__':
	print('reading...')
	alltext, accu_label, law_label, time_label = read_trainData('../final_all_data/exercise_contest/data_train.json')
	print('cut text...',print_mem())

	train_data = cut_text(alltext)
	joblib.dump(train_data,'inputseg/data_train.model', compress=3)

	print('train tfidf...',print_mem())
	tfidf = train_tfidf(train_data)
	
	vec = tfidf.transform(train_data)
	joblib.dump(vec, 'predictor/model/%s.model'% 'data_dev')

	print('accu SVC',print_mem())
	accu = train_SVC(vec, accu_label)
	print('law SVC',print_mem())
	law = train_SVC(vec, law_label)
	print('time SVC',print_mem())
	time = train_SVC(vec, time_label)
	
	print('saving model',print_mem())
	joblib.dump(tfidf, 'predictor/model/tfidf.model')
	joblib.dump(accu, 'predictor/model/accu.model')
	joblib.dump(law, 'predictor/model/law.model')
	joblib.dump(time, 'predictor/model/time.model')



