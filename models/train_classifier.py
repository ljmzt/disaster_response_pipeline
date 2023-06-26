import pandas as pd
import sqlite3
import sys
from model_utils import RandomForestClassifier_wrapper, ColumnTransformer_wrapper
from model_utils import prepare_df
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from flat_cv import FlatCV

def load_data(database_filepath):
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM messages', con=conn)
    conn.close()
    return df

def tokenize(text):
    ''' the tokenize is in the model pipeline '''
    pass


def build_model(cols):
    preprocess_pipe = ColumnTransformer([
        ('message', CountVectorizer(tokenizer=str.split,
                                    token_pattern=None,
                                    lowercase=False,
                                    min_df=10), 'message')
    ])
    
    clf_pipe = ColumnTransformer_wrapper([
        (col, 
         RandomForestClassifier_wrapper(class_weight='balanced', n_jobs=2), 
         None) for col in cols
    ])

    stack_pipe = ColumnTransformer_wrapper([
        (col, 
         LogisticRegression(class_weight='balanced', penalty='l1',solver="liblinear"), 
         None) for col in cols
    ])

    pipeline = Pipeline([
        ('preprocess', preprocess_pipe),
        ('clf', clf_pipe),
        ('stack', stack_pipe)
    ])

    param_grids = [
        None,
        {'min_samples_leaf': [1,2,3,5]},
        {'C': [1e-3,1e-2,1e-1,1,5,10]},    
    ]

    fclf = FlatCV(pipeline, cols, param_grids)

    return fclf


def evaluate_model(model, df_test, category_names):
    Y_pred = model.predict(df_test)
    Y_true = df_test[category_names].values
    for icol, category_name in enumerate(category_names):
        print(f'reporting {category_name}')
        print(classification_report(Y_pred[:,icol], Y_true[:,icol]))

def save_model(model, model_filepath):
    model.save(model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        df = load_data(database_filepath)
        df_train, df_test, category_names = prepare_df(df, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model(category_names)
        
        print('Training model...')
        model.fit(df_train, df_train[category_names].values)
        
        print('Evaluating model...')
        evaluate_model(model, df_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
