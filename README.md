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

app: contain the necessary py and html files to display the website  

### Summaries of Results

### Main technical points
1. 

### Libraries used
* python==3.9.13
* pandas==1.4.0
* numpy==1.23.3
* matplotlib==3.5.1
* seaborn==0.11.2
* scipy==1.9.1
* scikit-learn==1.0.2
