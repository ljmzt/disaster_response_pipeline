import sys
import pandas as pd
import sqlite3

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    messages.drop_duplicates(subset=['id'], keep=False, inplace=True)
    categories.drop_duplicates(subset=['id'], keep=False, inplace=True)
    df = pd.merge(left=messages, right=categories,
                  left_on='id', right_on='id',
                  how='inner')
    return df

def clean_data(df):
    tmp = df['categories'].str.split(';', expand=True)
    tmp.columns = [x.split('-')[0] for x in tmp.iloc[0].values]
    for col in tmp.columns:
        tmp[col] = tmp[col].str.split('-').str.get(1).astype(int)
    df = pd.concat((df.drop(columns=['categories']), tmp), axis=1)
    return df


def save_data(df, database_filename):
    conn = sqlite3.connect(database_filename)
    df.to_sql('messages', con = conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()
    return


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
