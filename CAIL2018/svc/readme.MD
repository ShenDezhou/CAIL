#1.Train model
run `python main.py`  

#2.Evaluate model
run `python evaluate.py`  
result: accusation score, law article score and penity time score.  
The small dataset's baseline(SVC) score is:  
71.83、68.79、47.83  
with improved dimensions of TFIDF, benchmark for the three tasks are:  
74.0, 70.37, 50.65

#3. Size of each dataset
dev: 46 dev.
excercise: 154592 training, 17131 validation, 32508 test.  
first_stage: 1710856 training, 217016 test. 
second_stage:  35922 test.  
unused: 748203.

#4. THE paper: CAIL2018: A Large-Scale Legal Dataset for Judgment Prediction
total: 2676075 cases, 183 law articles, 202 charges and prison term.  
top 10 charges covers 79.0% cases, last 10 charges covers 0.12% cases.  
***select 1710856 cases(first-stage-training) for training and 965219 cases for testing.(first-stage-test and unused)  

model benchmark:  

|MODEL| TASK | Acc %| Macro P% | Macro R%|
|---------|----|----|-----|-----|
|FastText| Charges |94.3| 50.9| 39.7|
|FastText| Relevant Articles |93.3| 45.8| 38.1|
|FastText| Terms of Penalty |74.6| 48.0| 24.5 |
|TFIDF+LinearSVC| Charges |94.0 | 73.9 | 56.2 |
|TFIDF+LinearSVC| Relevant Articles |92.9 | 71.8 | 52.4 |
|TFIDF+LinearSVC| Terms of Penalty |75.4 |75.4 | 46.1 |
|CNN| Charges |97.6| 37.0| 21.4|
|CNN| Relevant Articles |97.6| 37.4| 21.8|
|CNN| Terms of Penalty |78.2| 45.5| 36.1|

#5. THE paper: Overview of CAIL2018: Legal Judgment Prediction Competition.
Dataset: 5730302 in total.  
first stage:  1710856 training, 217016 test.  

***evaluating on second stage:  35922  (final_test)

metrics on a test set of 35922 documents:  

|Tasks |Law mi-f1 |Articles ma-f1|Charges mi-f1|  ma-f1   | Prison Terms Score (log distance)|
|----|----|-----|-----|-----|------|
|nevermore| 0.958| 0.781| 0.962| 0.836| 77.57|
|jiachx|0.952 |0.748 |0.958| 0.815| 69.64 |
|xlzhang|0.952 |0.760 |0.958 |0.811|69.64|
|HFL|0.953| 0.769 |0.958 |0.811| 77.70 |
|大师兄|0.945| 0.757| 0.951| 0.816|73.16 |
|安徽高院类案指引研发团队| 0.946| 0.756| 0.950 |0.803|72.24|
|AI judge|0.952 |0.766 |0.956 |0.811| - |
|只看看不说话|0.948 |0.738| 0.954| 0.801| 77.54|
|DG|0.945| 0.717| 0.949| 0.755|76.18|
|SXU AILAW|0.940| 0.728| 0.950| 0.791|76.49
|中电28所联合部落|0.934 |0.740| 0.937| 0.772|75.77

#6. 融入罪名关键词的法律判决预测多任务学习模型
CAIL2018 dataset-excercise: 154592 training, 17131 validation, 32508 test.    
preprocessed: use 17088 for dev, 154177 for training, 32433 for test.  

***evaluate on excercise dataset.(filtered excercise dataset)

|Tasks |Law mi-f1 |Articles ma-f1|Charges mi-f1|  ma-f1   |
|----|----|-----|-----|-----|
|MTL Gold|９６．４３１８|８４．４９５９|９７．５２９|９0.５．３３４５|

#7. Experiment log
|Tasks |Law Articles |Charges | Prison Terms Score|
|----|----|-----|-----|
|LinearSVC|
|LinearSVC-500000|0.7400 |0.7038| 0.5066 |
|SVC-poly-3| - | - | -|
|SVC-rfb| - | - | -|
|SVC-sigmoid| - | - | -|