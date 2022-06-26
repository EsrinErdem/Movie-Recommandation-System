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
from sklearn.neighbors import NearestNeighbors
import plotly.express as px

# Data
start = time.time()
final_table = pd.read_csv("data/full_final", engine='pyarrow')
final_table=final_table.iloc[:,1:]
#final_table.drop("Unnamed: 0", inplace=True, axis=1)
end1 = time.time()
print("csv1 read",(end1-start),'s')


# Initialisation de l'app
app2 = dash.Dash(__name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            meta_tags=[
                    {"name": "viewport", "content": "width=device-width, initial-scale=1"}
                ])

# layout -> Front-end
app2.layout = html.Div([
    html.H1('LES ANALYSES'),
    dcc.Checklist( 
    value='All',
    id='esrin'),
    dcc.Graph(id="graph1"),
    dcc.Graph(id="graph2"),
    dcc.Graph(id="graph3"),
    dcc.Graph(id="graph4"),
    dcc.Graph(id="graph5"),
    dcc.Graph(id="graph6"),
    dcc.Graph(id="graph7"),
    dcc.Graph(id="graph8"),
    dcc.Graph(id="graph9"),
    dcc.Graph(id="graph10"),
    dcc.Graph(id="graph11"),
    dcc.Graph(id="graph12"),
    dcc.Graph(id="graph13"),
    dcc.Graph(id="graph14"),
    dcc.Graph(id="graph15"),
    dcc.Graph(id="graph16"),
    dcc.Graph(id="graph17"),
    dcc.Graph(id="graph18"),
    dcc.Graph(id="graph19"),
    dcc.Graph(id="graph20"),
    dcc.Graph(id="graph21")])
#df=pd.DataFrame()
# back-end -> moteur de l'app
@app2.callback(
   Output(component_id='graph1', component_property='figure'),
   Input(component_id='esrin', component_property='value'),
   
)
def figure_update(df):
    df=final_table
    colors = ['Brown', 'orange', 'ff7f0e']
    fig=px.pie(df,values=df.dtypes.value_counts(), names=['int64', 'object', 'float64'],hole=.4)
    fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  marker=dict( colors=colors,line=dict(color='#000000', width=4)))
    fig.update_layout(title={"x":0.5, "xanchor":'center',"yanchor":'top'},
        title_text="<b>Types de Données<b>",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='<b>TYPES<b>', x=0.5, y=0.5, font_size=20, showarrow=False)],
    template="plotly_white")
    return fig
@app2.callback(
    Output(component_id='graph2', component_property='figure'),
    Input(component_id='esrin', component_property='value'),
    
)
def figure_update(df):
    df=final_table
    fig = px.histogram(df.birthYear, x='birthYear',nbins=40,
                    color_discrete_sequence=['brown'],
                   labels={'birthYear':'la date de naissance','count':'count'})
    fig.update_xaxes(title=' ',range=[1200,2100])
    fig.update_yaxes(title="<b>count<b>")
    fig.update_layout(title={"x":0.5, "xanchor":'center',"yanchor":'top'},
    title_text="<b>Histogramme de l'année de naissance<b>"
    ,template="plotly_white")
    return fig

# 3 eme grafique
@app2.callback(
    Output(component_id='graph3', component_property='figure'),
    Input(component_id='esrin', component_property='value'),
    
)
def figure_update(df):
    df=final_table
    fig = px.histogram(df.deathYear, x='deathYear',
                    nbins=30,
                    color_discrete_sequence=['brown'],
                    labels={'birthYear':'la date de naissance','count':'count'})
    fig.update_xaxes(title=' ',range=[1000,2100])
    fig.update_yaxes(title="<b>count<b>")
    fig.update_layout(title={"x":0.5, "xanchor":'center',"yanchor":'top'},
    title_text="<b>Histogramme de l'année de décès<b>",
    template="plotly_white")
    return fig

# 4eme grafique
@app2.callback(
    Output(component_id='graph4', component_property='figure'),
    Input(component_id='esrin', component_property='value'),
    
)
def figure_update(df):
    df=final_table
    fig = px.histogram(df.runtimeMinutes, x='runtimeMinutes',
                    nbins=40,height=600,width=1000,
                    color_discrete_sequence=['brown'],
                    labels={'runtimeMinutes':'la durée','count':'count'})
    fig.update_xaxes(title=' ',range=[0,400])
    fig.update_yaxes(title="<b>count<b>")
    fig.update_layout(title={"x":0.5, "xanchor":'center',"yanchor":'top'},
        title_text="<b>Histogramme de la durée du film<b>"
        ,template="plotly_white")
    return fig

# 5eme grafique
@app2.callback(
    Output(component_id='graph5', component_property='figure'),
    Input(component_id='esrin', component_property='value'),
    
)
def figure_update(df):
    df=final_table
    fig = px.histogram(df.averageRating, x='averageRating',nbins=20,
                    height=600,width=1000,
                    color_discrete_sequence=['brown'],
                   labels={'averageRating':'Note','count':'count'})
    fig.update_xaxes(title=' ',range=[0,10])
    fig.update_yaxes(title="<b>count<b>")
    fig.update_layout(title={"x":0.5, "xanchor":'center',"yanchor":'top'},
    title_text="<b>Histogramme des Notes Moyennes<b>"
    ,template="plotly_white")
    return fig
# 6eme grafique
@app2.callback(
    Output(component_id='graph6', component_property='figure'),
    Input(component_id='esrin', component_property='value'),
    
)
def figure_update(df):
    df=final_table
    fig = px.histogram(df.startYear, x='startYear',nbins=30,
                    height=600,width=1000,
                    color_discrete_sequence=['brown'],
                   labels={'startYear':'la date de film','count':'count'})
    fig.update_xaxes(title=' ',range=[1950,2030])
    fig.update_yaxes(title="<b>count<b>")
    fig.update_layout(title={"x":0.5, "xanchor":'center',"yanchor":'top'},
    title_text="<b>Les Dates des Films<b>",
    template="plotly_white")
    return fig

# 7eme grafique
@app2.callback(
    Output(component_id='graph7', component_property='figure'),
    Input(component_id='esrin', component_property='value'),
    
)
def figure_update(df):
    df=final_table
    fig = px.histogram(df.numVotes, x='numVotes',nbins=400,
                    height=600,width=1000,
                    color_discrete_sequence=['brown'],
                   labels={'numVotes':'Votes','count':'count'})
    fig.update_xaxes(title=' ',range=[0,200000])
    fig.update_yaxes(title="<b>count<b>")
    fig.update_layout(title={"x":0.5, "xanchor":'center',"yanchor":'top'},
    title_text="<b>Nombre De Votes Pour Les Films<b>",
    template="plotly_white")
    return fig

# 8eme grafique
@app2.callback(
    Output(component_id='graph8', component_property='figure'),
    Input(component_id='esrin', component_property='value'),
    
)
def figure_update(df):
    df=final_table
    fig = px.histogram(df.diffusion, x='diffusion',nbins=50,
                    height=600,width=1000,
                    color_discrete_sequence=['brown'],
                   labels={'diffusion':'le nombre de diffusion','count':'count'})
    fig.update_xaxes(title=' ',range=[0,160])
    fig.update_yaxes(title="<b>count<b>")
    fig.update_layout(title={"x":0.5, "xanchor":'center',"yanchor":'top'},
    title_text="<b>Nombres de Diffusion<b>",
    template="plotly_white")
    return fig

# 9eme grafique
df_genre=final_table.loc[:,['tconst','Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy',
       'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
       'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance', 'Sci-Fi',
       'Sport', 'Thriller', 'War', 'Western']]
df_genre.drop_duplicates(inplace=True)
list=[]
for i in df_genre.columns[1:]:
    list.append(round(df_genre[i].sum()*100/df_genre.shape[0],2))
pourcantage_genre=pd.DataFrame([list],columns=['Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy',
       'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
       'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance', 'Sci-Fi',
       'Sport', 'Thriller', 'War', 'Western'])
pourcantage_genre=pourcantage_genre.T
pourcantage_genre.reset_index(inplace=True)
pourcantage_genre.columns=['genres','frequencies']
pourcantage_genre=pourcantage_genre.head(23)
pourcantage_genre=pourcantage_genre.sort_values('frequencies',ascending=False)

@app2.callback(
    Output(component_id='graph9', component_property='figure'),
    Input(component_id='esrin', component_property='value'),
    
)
def figure_update(df):
    df=pourcantage_genre
    fig=px.bar(df ,x='genres',y='frequencies',
          height=600,width=1000,
        color_discrete_sequence=['brown'],
        labels={'genres':'Genres','frequencies':'La fréquence'})
    fig.update_yaxes(title='<b>les frequences des genres (%)<b>')
    fig.update_xaxes(title='<b>genres<b>')
    fig.update_layout(title={"x":0.5, "xanchor":'center',"yanchor":'top'},
    title_text="<b>REPARTITION PAR GENRE<b>",
    template="plotly_white")
    return fig
# 10 eme grafique
@app2.callback(
    Output(component_id='graph10', component_property='figure'),
    Input(component_id='esrin', component_property='value'),
    
)
def figure_update(df):
    df=final_table
    colors = [ 'brown', 'orange','yellow','tomato','cyan','darkgoldenrod']

    fig=px.pie(df["category"],
            values=final_table["category"].value_counts(),
            names=final_table["category"].value_counts().index)
    fig.update_traces( textfont_size=20,
                marker=dict( colors=colors,line=dict(color='#000000', width=2)))
    fig.update_layout(title={"x":0.5, "xanchor":'center',"yanchor":'top'},
                    title_text="<b>les frequencies des Professionnels<b>",
    template="plotly_white")
    return fig
# 11 eme grafique
actors_table = final_table[final_table["category"]=="actor"]
actorsName = pd.DataFrame(actors_table.primaryName.value_counts())
actorsName = actorsName.rename(columns={"primaryName" : "totalMovies"})
actorsName = actorsName.reset_index()
actorsName = actorsName.rename(columns={"index" : "name"})
actorsName=actorsName[actorsName['totalMovies']>=7].sort_values('totalMovies',ascending=False).head(40)
actorsName=actorsName.sort_values('totalMovies')
@app2.callback(
    Output(component_id='graph11', component_property='figure'),
    Input(component_id='esrin', component_property='value'),
    
)
def figure_update(df):
    df=actorsName

    fig = px.bar(df, y='name', x='totalMovies',
             height=1000,width=1000,
            color_discrete_sequence=['brown'],
            labels={'totalMovies':'le Nombre de Film','name':'name'})
    fig.update_yaxes(title=' ',range=[0,40])
    fig.update_xaxes(title=' ')
    fig.update_layout(title={"x":0.5, "xanchor":'center',"yanchor":'top'},
                  title_text="<b>NOMBRE DE FILMS PAR ACTEUR<b>",
                  template="plotly_white")
    return fig
# 12 eme grafique
actress_table = final_table[final_table["category"]=="actress"]
actressName = pd.DataFrame(actress_table.primaryName.value_counts())
actressName = actressName.rename(columns={"primaryName" : "totalMovies"})
actressName = actressName.reset_index()
actressName = actressName.rename(columns={"index" : "name"})
actressName=actressName[actressName['totalMovies']>=7].sort_values('totalMovies',ascending=False).head(40)
@app2.callback(
    Output(component_id='graph12', component_property='figure'),
    Input(component_id='esrin', component_property='value'),
    
)
def figure_update(df):
    df=actressName

    fig = px.bar(df.sort_values('totalMovies'), y='name', x='totalMovies',
             height=1000,width=1000,
            color_discrete_sequence=['brown'],
            labels={'totalMovies':'le Nombre de Film','name':'name'})
    fig.update_yaxes(title=' ',range=[0,40])
    fig.update_xaxes(title=' ')
    fig.update_traces(marker_color='brown')
    fig.update_layout(title={"x":0.5, "xanchor":'center',"yanchor":'top'},
                  title_text="<b>NOMBRE DE FILMS PAR ACTRESS<b>",
                  template="plotly_white")
    return fig

# 13 eme grafique
def age(by,sy):
    return by-sy

final_table['age at release'] = age(final_table['startYear'],final_table['birthYear'])

mask= (final_table['category']=='actor') | (final_table['category']=='actress')
final_table[mask]['age at release']
final_new=final_table[mask]
final_new1=final_new[['category','age at release']]
final_new2=pd.DataFrame(final_new1.groupby('category')['age at release'].mean())

@app2.callback(
    Output(component_id='graph13', component_property='figure'),
    Input(component_id='esrin', component_property='value'),
    
)
def figure_update(df):
    df=final_new2

    fig= px.bar(df,x =df.index, y="age at release",height=600,width=800,
           color_discrete_sequence=['brown'],
            labels={'category':'category','age at release':'âge'})
    fig.update_xaxes(title=' ')
    fig.update_yaxes(title=" ")
    fig.update_layout(title={"x":0.5, "xanchor":'center',"yanchor":'top'},
                  title_text="<b>Age Moyens des Acteurs / Actrices<b>",
                  template="plotly_white")
    return fig
# 14 eme grafique
df_run_rate = final_table.iloc[:,12:14]
df_run_rate["rate_class"] = pd.cut(df_run_rate["averageRating"],
                                    bins=[5,6,7,8,9,9.5,10],
                                    labels=["5 à 6","6 à 7","7 à 8","8 à 9", "9 à 9,5", "9,5 à 10"])
final_new_4=pd.DataFrame(round(df_run_rate.groupby('rate_class')['runtimeMinutes'].mean(),2))

@app2.callback(
    Output(component_id='graph14', component_property='figure'),
    Input(component_id='esrin', component_property='value'),
    
)
def figure_update(df):
    df=final_new_4

    fig= px.bar(df,x =df.index, y='runtimeMinutes',height=600,width=800,
                labels={'rate_class':'rate','runtimeMinutes':'la durée de film'})
    fig.update_yaxes(title="<b>Durée moyenne<b>")
    fig.update_xaxes(title='<b>Note<b>')
    fig.update_traces(marker_color='brown')
    fig.update_layout(title={"x":0.5, "xanchor":'center',"yanchor":'top'},
                    title_text="<b>Durée VS Note<b>",
                    template="plotly_white")
    return fig

# 15 eme grafique
liste_genres = ['Action', 'Adult',
       'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary',
       'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Musical',
       'Mystery', 'News', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War',
       'Western']
genre_rating_dict = {}
for genre in liste_genres:
    df = final_table[final_table[genre]==1]
    genre_rating_dict[genre] = round(df["averageRating"].mean(),1)

df_ratings_genre = pd.DataFrame.from_dict(genre_rating_dict, orient='index', columns=['avg_rating'])
df_ratings_genre = df_ratings_genre.reset_index()
df_ratings_genre = df_ratings_genre.rename(columns={'index':'genre'})
df_ratings_genre.sort_values('avg_rating', axis=0,  ascending=False, inplace=True)
df_ratings_genre

@app2.callback(
    Output(component_id='graph15', component_property='figure'),
    Input(component_id='esrin', component_property='value'),
    
)
def figure_update(df):
    df=df_ratings_genre

    fig= px.bar(df,x ='genre', y='avg_rating',height=500,width=800,
                labels={'rate_class':'genre','avg_rating':'note'})
    fig.update_yaxes(title="<b>la moyenne note<b>")
    fig.update_xaxes(title='<b>genre<b>')
    fig.update_traces(marker_color='brown')
    fig.update_layout(title={"x":0.5, "xanchor":'center',"yanchor":'top'},
                    title_text="<b>Note moyenne par genre <b>",
                    template="plotly_white")
    return fig

# 16 eme grafique
final_table["genres2"] = final_table["genres"].apply(lambda x: eval(x))
df_trans = final_table.explode('genres2')
df_rep_genres = round(df_trans["genres2"].value_counts(normalize=True)*100,2)
df_rep_genres = df_rep_genres.to_frame().reset_index().rename(columns={"index":"genre","genres2":"percent"})
df_rep_genres

@app2.callback(
    Output(component_id='graph16', component_property='figure'),
    Input(component_id='esrin', component_property='value'),
    
)
def figure_update(df):
    df=df_rep_genres

    fig= px.bar(df,x ='genre', y='percent',height=800,width=1000,
            labels={'genre':'Genre','percent':'Percent'})
    fig.update_yaxes(title="<b>Part du Genre (en %)<b>")
    fig.update_xaxes(title='<b>Genre<b>')
    fig.update_traces(marker_color='brown')
    fig.update_layout(title={"x":0.5, "xanchor":'center',"yanchor":'top'},
                  title_text='<b>Répartition des Genres<b>',
                  template="plotly_white")
    return fig

# 17 eme grafique
df_year_runtime_genre = final_table.iloc[:,11:19]
df_year_runtime_genre.drop(columns=['averageRating', 'numVotes',
       'region', 'language', 'diffusion'], inplace=True)
df_year_runtime_genre["genres"] = df_year_runtime_genre["genres"].apply(lambda x: eval(x))
df_trans = df_year_runtime_genre.explode('genres')
df_trans_genre = df_trans.groupby(["startYear","genres"]).mean().reset_index()
df_trans_run = df_trans.groupby("startYear").mean().reset_index()

@app2.callback(
    Output(component_id='graph17', component_property='figure'),
    Input(component_id='esrin', component_property='value'),
    
)
def figure_update(df):
    df=df_trans_genre

    fig= px.scatter(df,x='genres', y='runtimeMinutes',height=800,width=1000
            ,animation_frame='startYear',
            labels={'startYear':'Year','runtimeMinutes':'Durée'},size='runtimeMinutes',
               size_max=30,color_discrete_sequence=['brown'])
    fig.update_xaxes(title="")
    fig.update_yaxes(title='<b>La Moyenne Durée<b>',range=[0,200])
    fig.update_traces(marker_color='brown')
    fig.update_layout(title={"x":0.5, "xanchor":'center',"yanchor":'top'},
                  title_text='<b>Répartition des Genres Par Année<b>')
    return fig

# 18 eme grafique
df_top10=final_table[['startYear','primaryTitle','averageRating']].sort_values(['startYear','averageRating'],ascending=[True,False])
df_top10.drop_duplicates(inplace=True)
df_top10.reset_index(drop=True,inplace=True)
df_top10.sort_values(['startYear','averageRating'], ascending=[True,False],inplace=True)

@app2.callback(
    Output(component_id='graph18', component_property='figure'),
    Input(component_id='esrin', component_property='value'),
    
)
def figure_update(df):
    df=df_top10

    fig= px.bar(df,x='primaryTitle', y='averageRating',height=500,width=1000
            ,animation_frame='startYear',
            labels={'startYear':'Year','primaryTitle':'Film','averageRating':'note'},color_discrete_sequence=['brown'])
    fig.update_xaxes(title=" ", range=[0,12])
    fig.update_yaxes(title=' ',range=[0,10])
    fig.update_traces(marker_color='brown')
    fig.update_layout(title={"x":0.5, "xanchor":'center',"yanchor":'top'},
                  title_text='<b>LES MEILLEURS FILMS PAR CHAQUE ANNEE<b>')
    return fig

# 19 eme grafique
df_language=final_table[['tconst','language']].drop_duplicates()
df_language.reset_index(drop=True, inplace=True)

@app2.callback(
    Output(component_id='graph19', component_property='figure'),
    Input(component_id='esrin', component_property='value'),
    
)
def figure_update(df):
    df=df_language

    colors = [ 'brown', 'orange', 'ff7f0e']

    fig=px.pie(df,
            values=df["language"].value_counts(),
            names=df["language"].unique())
    fig.update_traces( textfont_size=20,
                marker=dict( colors=colors,line=dict(color='#000000', width=2)))
    fig.update_layout(title={"x":0.5, 'y':1.0,"xanchor":'center',"yanchor":'top'},
                    title_text="<b>Les Langues Disponibles des Films<b>")
    return fig

# 20 eme grafique
df_diffusion=final_table.sort_values('diffusion', ascending=False)
df_diffusion=pd.DataFrame(df_diffusion.groupby(['tconst','primaryTitle'])['diffusion'].unique())
df_diffusion.reset_index(['tconst','primaryTitle'],inplace=True)
df_diffusion=df_diffusion.explode('diffusion')
df_diffusion.sort_values('diffusion', ascending=False, inplace=True)

@app2.callback(
    Output(component_id='graph20', component_property='figure'),
    Input(component_id='esrin', component_property='value'),
    
)
def figure_update(df):
    df=df_diffusion

    fig= px.bar(df.head(40),x='primaryTitle', y='diffusion',height=800,width=1000,
            labels={'primaryTitle':'Film','diffusion':'Diffusion'})
    fig.update_xaxes(title=" ")
    fig.update_yaxes(title='<b>le nombre de copies<b>',range=[0,200])
    fig.update_traces(marker_color='brown')
    fig.update_layout(title={"x":0.5, "xanchor":'center',"yanchor":'top'},
                  title_text='<b>MOST COPIED MOVIES<b>')
    return fig

# 21 eme grafique
df_rating=final_table[['tconst','primaryTitle','averageRating','numVotes']].sort_values(['averageRating','numVotes'],ascending=False)
df_rating=pd.DataFrame(df_rating.groupby(['tconst','primaryTitle','averageRating'])['numVotes'].unique())
df_rating.reset_index(['tconst','primaryTitle','averageRating'],inplace=True)
df_rating=df_rating.explode('numVotes')
df_rating.sort_values(by=['averageRating','numVotes'],axis=0, ascending=[False,False],inplace=True)

@app2.callback(
    Output(component_id='graph21', component_property='figure'),
    Input(component_id='esrin', component_property='value'),
    
)
def figure_update(df):
    df=df_rating

    fig= px.bar(df.head(40),x='primaryTitle', y='averageRating',height=800,width=1000,
            labels={'primaryTitle':'Film','averageRating':'Note'})
    fig.update_xaxes(title=" ")
    fig.update_yaxes(title='<b>Note<b>')
    fig.update_traces(marker_color='brown')
    fig.update_layout(title={"x":0.5, "xanchor":'center',"yanchor":'top'},
                  title_text='<b>TOP RATED MOVIES<b>')
    return fig



if __name__ == '__main__':
    app2.run_server(debug=True,port='8000') 