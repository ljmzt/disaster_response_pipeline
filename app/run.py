# see this for a fix for the plotly not rendering issue
# https://knowledge.udacity.com/questions/983227

import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import pickle
import sqlite3
from model_utils import prepare_df

app = Flask(__name__)

# load data
conn = sqlite3.connect('../data/data_for_ml.db')
df = pd.read_sql('SELECT * FROM messages', con=conn)
cols = df.drop(columns=['child_alone']).columns[4:]

# load model
with open("../models/model_individual_stack.pickle", 'rb') as fid:
    model = pickle.load(fid)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    tmp = df.iloc[:,4:]
    category_names = tmp.columns
    category_counts = (tmp>0).sum(axis=0)
    category_frac = (tmp>0).mean(axis=0)*100
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [

# graph 1
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

# graph 2
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'automargin': True 
                }
            }
        },

# graph 3
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_frac
                )
            ],

            'layout': {
                'title': 'Percentage of Message Categories',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Category",
                    'automargin': True
                }
            }
        }

    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 
    query_saved = query
    query = pd.DataFrame(data=[[query,query]], columns=['message','original'])
    query = prepare_df(query, predict=True)

    # use model to predict classification for query
    classification_labels = model.predict(query)[0]
    classification_results = dict(zip(cols, classification_labels))
    print(classification_labels)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query_saved,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
