# Disaster Response Pipeline Project

### Purpose 
This is Udacity Data Scientist Nano-degree Project. The purpose is 1) to learn how to set up pipeline 2) to classify disaster message into multilabel categories. The raw data is provided from Appen (formally Figure 8).

### Main Files
data/process_data.py: script to perform the ETL task.  
data/disaster_categories.csv, disaster_messages.csv: raw data  
data/data_for_ml.db: output from the ETL process  

models/train_classifier.py: script to generate the file model  
models/test-preprocess.ipynb, fit-model-fit-individual-v2.ipynb: two simpler models; see technical points below  
models/compare.ipynb: compare different models  
models/flat_cv.py flat_classifier.py: script for flat classifier and CV  
models/model_utils.py, cv_result_analyzer.py: helper functions  

app/: contain the necessary py and html files to display the website  

The final model is too slow to upload to github for me, so I uploaded it to [google drive](https://drive.google.com/file/d/1AXjRJpRY_uk9FHfmOUvXs_H8eXwZJ8SF/view?usp=drive_link). Please put in under models/ folder if you would like to run the website. 

### Main technical points in modeling
1. For the 'related' label, I combine class 0 and 2 into one class. There're actually only very few class 2 records, so should be fine. Now the problem is a pretty standard multioutput/multilabel binary classification problem

2. Existing multioutput libary in sklearn doesn't allow fine tuning for each label, so I have to code it up myself. The key codes are flat_classifier.py and fit-model-fit-individual-v2.ipynb. This can improve the macro-f1 score from 0.473 to 0.483. I also tested adding an L1-norm Logistic Regression after the classifier. It is not quite useful (0.484 vs 0.484), but it does improve the f1-score on some difficult labels (e.g. tools, shops). See compare.ipynb for detail comparison. 

3. In the test-preprocess.ipynb, I compare somewhat different preprocessing strategies. To my surprise, doing TfIdf is not useful at all. I also tested adding a language detector, but that is also not that useful. So I stick with countvectorize, with some less frequent words removed. See the notebook for details.

4. I coded up a cv_result_analyzer, which can be a generalized code to examine the cv_results.

### Summaries of Results
I managed to obtain an macro f1-score of 0.484 in the final model.

### Libraries used
* python==3.9.13
* pandas==1.4.0
* numpy==1.23.3
* matplotlib==3.5.1
* seaborn==0.11.2
* scipy==1.9.1
* scikit-learn==1.0.2
