from ast import In
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from collections import deque 
import random
from datetime import datetime
import sqlite3
import dash_bootstrap_components as dbc
import pandas as pd 
import numpy as np 
import time as time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# DATA
movie_user_likes = "Quantum of Solace"

print('damn')

start = time.time()

df = pd.read_csv('data/caea_4models',engine='pyarrow')

end1 = time.time()
print("csv1 read",(end1-start),'s')
cv = CountVectorizer()



##########################################################################################################################
#We create matrix of binary lists and compare each movie with them 

count_matrix = cv.fit_transform(df["combined_features"])

#We find the index with the title entered by the consumer
def get_index_from_title(primaryTitle):
    return df[df.primaryTitle == primaryTitle].index.values[0]
movie_index = get_index_from_title(movie_user_likes)

#We use the index to create a list of similar movies with cosine_sim function which will compare the binary lists
cosine_sim = cosine_similarity(count_matrix)
similar_movies = list(enumerate(cosine_sim[movie_index]))

#We sort by the best scores
sorted_similar_movies = sorted(similar_movies, key=lambda x:x[1], reverse=True)


##########################################################################################################################

count_matrix_2 = cv.fit_transform(df["combined_features_2"])

#We use the index to create a list of similar movies with cosine_sim function which will compare the binary lists
cosine_sim_2 = cosine_similarity(count_matrix_2)
similar_movies_2 = list(enumerate(cosine_sim_2[movie_index]))

#We sort by the best scores
sorted_similar_movies_2 = sorted(similar_movies_2, key=lambda x:x[1], reverse=True)

##########################################################################################################################
count_matrix_3 = cv.fit_transform(df["combined_features_3"])

#We use the index to create a list of similar movies with cosine_sim function which will compare the binary lists
cosine_sim_3 = cosine_similarity(count_matrix_3)
similar_movies_3 = list(enumerate(cosine_sim_3[movie_index]))

#We sort by the best scores
sorted_similar_movies_3 = sorted(similar_movies_3, key=lambda x:x[1], reverse=True)

##########################################################################################################################
count_matrix_4 = cv.fit_transform(df["combined_features_4"])

#We use the index to create a list of similar movies with cosine_sim function which will compare the binary lists
cosine_sim_4 = cosine_similarity(count_matrix_4)
similar_movies_4 = list(enumerate(cosine_sim_4[movie_index]))

#We sort by the best scores
sorted_similar_movies_4 = sorted(similar_movies_4, key=lambda x:x[1], reverse=True)

##########################################################################################################################

# initialisation app
app = dash.Dash(__name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            meta_tags=[
                    {"name": "viewport", "content": "width=device-width, initial-scale=1"}
                ])


# Layout
app.layout = html.Div([

    dbc.Row([
        html.Div([
            html.Img(src='assets\Logonetflix.png', className='logo'),
            html.H1("Bienvenue sur l'application de recommandation de votre cinéma préféré"),
            html.P(["Caeaflix est heureux de vous présenter ce système de recommandation.",html.Br(),"Vous avez apprécié un film ? Un réalisateur ? Un studio de production ?",html.Br(),"Entrez son titre dans la barre de recherche et découvrez d'autre propositions de voyage ! "],className='intro')
        ],className='header'),
        html.Div([
                dbc.Input('Entrez le titre de votre film', className='input'),
                html.H2('Vos Recommandations :'),
                
            ],className='Title'),
        html.Div([
            html.H2('Du même genre'),
                html.Div([
                    
                    html.Div([
                        html.Img(src=(df.loc[sorted_similar_movies[1:4][0][0],['movies_poster']][0]), className='Posters'),
                        html.P(f"{df.loc[[sorted_similar_movies[1:4][0][0]]][['primaryTitle']].values[0][0]}",className='Infos'),
                        html.P(f"{df.loc[[sorted_similar_movies[1:4][0][0]]][['movies_overview']].values[0][0]}",className='overview'),
                    ],className='Recommandation'),

                    html.Div([
                        html.Img(src=(df.loc[sorted_similar_movies[1:4][1][0],['movies_poster']][0]), className='Posters'),
                        html.P(f"{df.loc[[sorted_similar_movies[1:4][1][0]]][['primaryTitle']].values[0][0]}",className='Infos'),
                        html.P(f"{df.loc[[sorted_similar_movies[1:4][1][0]]][['movies_overview']].values[0][0]}",className='overview'),
                    ],className='Recommandation'),

                    html.Div([
                        html.Img(src=(df.loc[sorted_similar_movies[1:4][2][0],['movies_poster']][0]), className='Posters'),
                        html.P(f"{df.loc[[sorted_similar_movies[1:4][2][0]]][['primaryTitle']].values[0][0]}",className='Infos'),
                        html.P(f"{df.loc[[sorted_similar_movies[1:4][2][0]]][['movies_overview']].values[0][0]}",className='overview'),
                    ],className='Recommandation')
            ],className='divcentrer')
        ],className='menu-container'),


        html.Div([
            html.H2('Du même Réalisateur'),
                html.Div([
                    
                    html.Div([
                        html.Img(src=(df.loc[sorted_similar_movies_2[1:4][0][0],['movies_poster']][0]), className='Posters'),
                        html.P(f"{df.loc[[sorted_similar_movies_2[1:4][0][0]]][['primaryTitle']].values[0][0]}",className='Infos'),
                        html.P(f"{df.loc[[sorted_similar_movies_2[1:4][0][0]]][['movies_overview']].values[0][0]}",className='overview'),
                    ],className='Recommandation'),

                    html.Div([
                        html.Img(src=(df.loc[sorted_similar_movies_2[1:4][1][0],['movies_poster']][0]), className='Posters'),
                        html.P(f"{df.loc[[sorted_similar_movies_2[1:4][1][0]]][['primaryTitle']].values[0][0]}",className='Infos'),
                        html.P(f"{df.loc[[sorted_similar_movies_2[1:4][1][0]]][['movies_overview']].values[0][0]}",className='overview'),
                    ],className='Recommandation'),

                    html.Div([
                        html.Img(src=(df.loc[sorted_similar_movies_2[1:4][2][0],['movies_poster']][0]), className='Posters'),
                        html.P(f"{df.loc[[sorted_similar_movies_2[1:4][2][0]]][['primaryTitle']].values[0][0]}",className='Infos'),
                        html.P(f"{df.loc[[sorted_similar_movies_2[1:4][2][0]]][['movies_overview']].values[0][0]}",className='overview'),
                    ],className='Recommandation')
            ],className='divcentrer')
        ],className='menu-container'),

        html.Div([
            html.H2('Du même Studio'),
                html.Div([
                    
                    html.Div([
                        html.Img(src=(df.loc[sorted_similar_movies_3[4:7][0][0],['movies_poster']][0]), className='Posters'),
                        html.P(f"{df.loc[[sorted_similar_movies_3[4:7][0][0]]][['primaryTitle']].values[0][0]}",className='Infos'),
                        html.P(f"{df.loc[[sorted_similar_movies_3[4:7][0][0]]][['movies_overview']].values[0][0]}",className='overview'),
                    ],className='Recommandation'),

                    html.Div([
                        html.Img(src=(df.loc[sorted_similar_movies_3[4:7][1][0],['movies_poster']][0]), className='Posters'),
                        html.P(f"{df.loc[[sorted_similar_movies_3[4:7][1][0]]][['primaryTitle']].values[0][0]}",className='Infos'),
                        html.P(f"{df.loc[[sorted_similar_movies_3[4:7][1][0]]][['movies_overview']].values[0][0]}",className='overview'),
                    ],className='Recommandation'),

                    html.Div([
                        html.Img(src=(df.loc[sorted_similar_movies_3[4:7][2][0],['movies_poster']][0]), className='Posters'),
                        html.P(f"{df.loc[[sorted_similar_movies_3[4:7][2][0]]][['primaryTitle']].values[0][0]}",className='Infos'),
                        html.P(f"{df.loc[[sorted_similar_movies_3[4:7][2][0]]][['movies_overview']].values[0][0]}",className='overview'),
                    ],className='Recommandation')
            ],className='divcentrer')
        ],className='menu-container'),
         html.Div([
            html.H2('Avec les mêmes acteurs/actrices'),
                html.Div([
                    
                    html.Div([
                        html.Img(src=(df.loc[sorted_similar_movies_4[4:7][0][0],['movies_poster']][0]), className='Posters'),
                        html.P(f"{df.loc[[sorted_similar_movies_4[4:7][0][0]]][['primaryTitle']].values[0][0]}",className='Infos'),
                        html.P(f"{df.loc[[sorted_similar_movies_4[4:7][0][0]]][['movies_overview']].values[0][0]}",className='overview'),
                    ],className='Recommandation'),

                    html.Div([
                        html.Img(src=(df.loc[sorted_similar_movies_4[4:7][1][0],['movies_poster']][0]), className='Posters'),
                        html.P(f"{df.loc[[sorted_similar_movies_4[4:7][1][0]]][['primaryTitle']].values[0][0]}",className='Infos'),
                        html.P(f"{df.loc[[sorted_similar_movies_4[4:7][1][0]]][['movies_overview']].values[0][0]}",className='overview'),
                    ],className='Recommandation'),

                    html.Div([
                        html.Img(src=(df.loc[sorted_similar_movies_4[4:7][2][0],['movies_poster']][0]), className='Posters'),
                        html.P(f"{df.loc[[sorted_similar_movies_4[4:7][2][0]]][['primaryTitle']].values[0][0]}",className='Infos'),
                        html.P(f"{df.loc[[sorted_similar_movies_4[4:7][2][0]]][['movies_overview']].values[0][0]}",className='overview'),
                    ],className='Recommandation')
            ],className='divcentrer')
        ],className='menu-container'),
    ],className='Row'),
],className='page')


print('finito')

# Back-end
"""
@cosine.callback(
   Output(component_id='dumemegenre', component_property='children'),
   Input(component_id='esrin', component_property='value'),)
"""


if __name__ == '__main__' :
    app.run_server(debug=True)