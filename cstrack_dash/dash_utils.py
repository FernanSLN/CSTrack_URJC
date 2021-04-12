import generate_utils as utils
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import re
import networkx as nx
import pymongo
import plotly.graph_objects as go
import networkit as nk
from functools import lru_cache


# Custom function to create an edge between node x and node y, with a given text and width
def make_edge(x, y, text):
    return go.Scatter(x=x,
                      y=y,
                      line=dict(width=1,
                                color='cornflowerblue'),
                      hoverinfo='text',
                      text=([text]),
                      mode='lines')


def get_graph_figure(G, i = 0):
    print("hello")
    pos = nx.spring_layout(G, iterations=10)
    print("Positions given")
    for n, p in pos.items():
        G.nodes[n]['pos'] = p

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    count = 0
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
        count += 1

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='RdBu',
            reversescale=True,
            color=[],
            size=15,
            colorbar=dict(
                thickness=10,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=0)))
    count = 0
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        count += 1

    for node, adjacencies in enumerate(G.adjacency()):
        node_trace['marker']['color'] += tuple([len(adjacencies[1])])
        node_info = str(adjacencies[0]) + ' # of connections: ' + str(len(adjacencies[1]))
        node_trace['text'] += tuple([node_info])

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Community ' + str(i + 1),
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="No. of connections",
                            showarrow=False,
                            xref="paper", yref="paper")],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    print("Finish creating figure")
    return fig


def get_community_graph(g, community, i=0):
    c = nx.DiGraph()
    for node in community[i]:
        list_edges = g.edges(node)
        list_edges = [edge for edge in list_edges if edge[1] in community[i]]
        c.add_edges_from(list_edges)
    return c

@lru_cache(maxsize=None)
def get_communities(g):
    n_g = nk.nxadapter.nx2nk(g)
    idmap = dict((u, id) for (id, u) in zip(g.nodes(), range(g.number_of_nodes())))
    communities = nk.community.detectCommunities(n_g)
    list_communities = []
    for i in range(0, communities.numberOfSubsets()):
        list_members = []
        for member in communities.getMembers(i):
            list_members.append(idmap[member])
        if len(list_members) > 5:
            list_communities.append(list_members)
    return list_communities

def get_n_tweets(df):
    base_date = df.iloc[0]["Date"]
    base_count = df.iloc[0]["Number"]
    result = []
    for i, data in df.iterrows():
        count = data["Number"] - base_count
        base_count = data["Number"]
        result.append({"Date": data["Date"], "Number": count})
    return pd.DataFrame(result)


def acumulate_retweets(df):
    base_count = df.iloc[0]["Number"]
    result = []
    for i, data in df.iterrows():
        count = data["Number"] + base_count
        base_count = count
        result.append({"Date": data["Date"], "Number": count})
    return pd.DataFrame(result)


def get_figure(df):
    fig = px.bar(df, x="Hashtags", y="Count", barmode="group")
    return fig


def get_temporal_figure(df, n_hashtags=2):
    fig = px.line(df, x='date', y=df.columns[:n_hashtags])
    fig.update_layout(xaxis_tick0=df['date'][0], xaxis_dtick=86400000 * 15)
    return fig


def get_cstrack_graph(df, type, title):
    df_retweets = df[df["Type"] == type]
    if type == "Retweets":
        df_retweets = df_retweets.groupby(["Date"], as_index=False)["Number"].sum()
        df_retweets["Date"] = pd.to_datetime(df_retweets['Date'], format="%d/%m/%Y")
        df_retweets = df_retweets.sort_values(by="Date")
        df_accumulated = acumulate_retweets(df_retweets)
        fig = px.line(df_accumulated, x="Date", y="Number", title=title)
        fig.add_trace(go.Scatter(x=df_retweets["Date"].tolist(), y=df_accumulated["Number"].tolist(),
                                 mode="markers", textposition="top center", name="Retweets per day",
                                 text=df_retweets["Number"].tolist()))
    elif type == "Tweets":
        print(df_retweets.dtypes)
        df_retweets["Date"] = pd.to_datetime(df_retweets['Date'], format="%d/%m/%Y").dt.date
        df_retweets = df_retweets.sort_values(by="Date")
        df_retweets = get_n_tweets(df_retweets.drop_duplicates(subset=["Date"])).iloc[1:]
        fig = px.line(df_retweets, x="Date", y="Number", title=title)
    if type == "Followers":
        single_follow_count = get_n_tweets(df_retweets)
        fig = px.line(df_retweets, x="Date", y="Number", title=title)
        fig.add_trace(go.Scatter(x=df_retweets["Date"].tolist(), y=df_retweets["Number"].tolist(),
                                 mode="markers+text", textposition="top center", name="New followers",
                                 text=single_follow_count["Number"].tolist()))

    return fig


def get_df_ts(df, days, elements):
    numHashtag = []
    for hashtag in elements[:2]:
        numPerDay = []
        for day in days:
            dfOneDay = df[df['Fecha'] == day]
            count = dfOneDay['Texto'].str.contains(hashtag, case=False).sum()
            numPerDay.append(count)
        numHashtag.append(numPerDay)
    ts_df = pd.DataFrame()
    for i in range(0, len(numHashtag)):
        ts_df[elements[i]] = numHashtag[i]
    ts_df = ts_df.assign(date=days)
    return ts_df


def get_rt_hashtags(df, k=None, stop=None, n_hashtags=10):
    listHashtagsRT2 = utils.get_hashtagsRT(df, keywords=k, stopwords=stop)
    edges = utils.get_edgesHashRT(listHashtagsRT2)
    # Con las stopwords eliminamos el bot:
    sortedNumberHashtags, sortedHashtagsmain = utils.prepare_hashtags(edges)
    df_hashtags = pd.DataFrame(list(zip(sortedHashtagsmain, sortedNumberHashtags)), columns=["Hashtags", "Count"])
    return df_hashtags


def get_all_hashtags(df, k=None, stop=None, n_hashtags=10):
    hashmain = utils.get_hashtagsmain(df, keywords=k)
    edges = utils.get_edgesMain(hashmain)
    # Con las stopwords eliminamos el bot:
    sortedNumberHashtags, sortedHashtagsmain = utils.prepare_hashtagsmain(edges, stopwords=['airpollution', 'luftdaten',
                                                                                            'fijnstof', 'waalre', 'pm2',
                                                                                            'pm10'])
    df_hashtags = pd.DataFrame(list(zip(sortedHashtagsmain, sortedNumberHashtags)), columns=["Hashtags", "Count"])
    return df_hashtags


def get_all_temporalseries(df, k=None, stop=None):
    df = df[['Usuario', 'Texto', 'Fecha']].copy()
    df = df.dropna()
    df = df[df['Fecha'].str.match('[0-9][0-9]/[0-9][0-9]/[0-9][0-9][0-9][0-9]\s[0-9]')]
    df["Fecha"] = pd.to_datetime(df['Fecha'], format="%d/%m/%Y %H:%M").dt.date
    dias = utils.getDays(df)
    listHt = utils.get_hashtagsmain(df, keywords=k)
    edges = utils.get_edgesMain(listHt)
    sortedNH, sortedMH = utils.prepare_hashtagsmain(edges, stopwords=utils.botwords)
    return df, dias, sortedMH


def get_rt_temporalseries(df, k=None, stop=None):
    df = df[['Usuario', 'Texto', 'Fecha']].copy()
    df = df.dropna()
    df = df[df['Fecha'].str.match('[0-9][0-9]/[0-9][0-9]/[0-9][0-9][0-9][0-9]\s[0-9]')]
    df["Fecha"] = pd.to_datetime(df['Fecha'], format="%d/%m/%Y %H:%M").dt.date
    dias = utils.getDays(df)
    listHt = utils.get_hashtagsRT(df, keywords=k)
    edges = utils.get_edgesHashRT(listHt)
    sortedNH, sortedMH = utils.prepare_hashtags(edges)
    return df, dias, sortedMH


def wordcloudmain(df, keywords=None, stopwords=None, interest=None):
    hashtags = []
    stop_words = ['citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana', 'CitizenScience']
    df = df[['Usuario', 'Texto']]
    df = df.dropna()
    idx = df[df['Texto'].str.match('RT @')]
    df = df.drop(idx.index)
    subset = df['Texto']
    for row in subset:
        match = re.findall('#(\w+)', row.lower())
        for hashtag in match:
            hashtags.append(hashtag)
    unique_string = (' ').join(hashtags)
    wordcloud = WordCloud(width=900, height=600, background_color='white', stopwords=stop_words,
                          min_font_size=10, max_words=10405, collocations=False, colormap='winter').generate(
        unique_string)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig("./assets/wc2.png")


def get_graph_rt(df):
    retweetList = utils.get_retweets(df)
    retweetEdges = utils.get_edges(retweetList)
    G = nx.Graph()
    G.add_edges_from(retweetEdges)
    return G


def get_degrees(df):
    from operator import itemgetter
    import datetime
    start = datetime.datetime.now()
    retweetList = utils.get_retweets(df)
    retweetEdges = utils.get_edges(retweetList)
    G = nx.DiGraph()
    G.add_edges_from(retweetEdges)
    print("FINALIZA,", datetime.datetime.now() - start)
    return utils.get_degrees(G)


def get_twitter_info_df():
    db = pymongo.MongoClient(host="127.0.0.1", port=27017)
    twitter_data = db["cstrack"]["cstrack_stats"]
    twitter_dict_list = list(twitter_data.find())
    df = pd.DataFrame(twitter_dict_list)
    df = df.dropna()
    df["Date"] = pd.to_datetime(df['Date'], format="%d/%m/%Y %H:%M", errors="ignore")
    df["Date"] = pd.to_datetime(df['Date'], format="%d/%m/%Y", errors="ignore")
    return df


def get_controls_community(communities):
    dropdown_options = []
    for i in range(0, len(communities)):
        dropdown_options.append({"label": str(i), "value": i})
    controls = dbc.Form(
        [
            dbc.FormGroup(
                [
                    dbc.Label("Community:"),
                    dcc.Dropdown(
                        id="com_number",
                        options=dropdown_options,
                        value=0,
                        clearable=False,
                        style = {"margin-left": "2px"}
                    )
                ],
                className="mr-3",
            ),
            dbc.FormGroup(
                [
                    dbc.Label("Algorithm:"),
                    dcc.Dropdown(
                        id="com_algorithm",
                        options=[{"label": "Louvain", "value": "louvain"}, {"label": "Label propagation", "value": "propagation"}],
                        value="louvain",
                        clearable=False,
                        style={"width": "200px", "margin-left": "2px"}
                    )
                ],
            ),
        ],
        inline=True
    )
    return controls
@lru_cache(maxsize=None)
def get_controls_activity():
    controls = dbc.Form(
        [
            dbc.FormGroup(
                [
                    dbc.Label("Activity:"),
                    dcc.Dropdown(
                        id="activity_type",
                        options=[{"label": "Tweets", "value": "tweets"}, {"label": "Followers", "value": "followers"}],
                        value="tweets",
                        clearable=False,
                        style={"width": "200px", "margin-left": "2px"}
                    )
                ],
            ),
        ],
        inline=True
    )
    return controls
@lru_cache(maxsize=None)
def get_controls_rt(number_id, keyword_id):
    controls = dbc.Form(
        [
            dbc.FormGroup(
                [
                    dbc.Label("Number hashtags:"),
                    dbc.Input(id=number_id, n_submit=0, type="number", value=10, debounce=True),
                ],
                className="mr-3",
            ),
            dbc.FormGroup(
                [
                    dbc.Label("Keywords:"),
                    dbc.Input(id=keyword_id, n_submit=0, type="text", value="", debounce=True),
                ],
                className="mr-3"
            ),
        ],
        inline=True
    )
    return controls

@lru_cache(maxsize=None)
def set_loading(controls, dcc_graph):
    SPINER_STYLE = {
        "margin-top": "25%",
        "width": "99%",
        "height": "20vh",
        "text-align": "center",
        "font-size": "50px",
        "margin-left": "1%",
        "z-index": "1000"
    }
    loading = dcc.Loading(
        # style={"height":"200px","font-size":"100px","margin-top":"500px", "z-index":"1000000"},
        style=SPINER_STYLE,
        color="#000000",
        id="loading-1",
        type="default",
        children=html.Div(id="loading-output", children=[
            dbc.Row(controls, justify="center"),
            dbc.Row(
                children=[dcc_graph], justify="center"
            )
        ])
    ),
    return loading

def get_map_df():
    con = pymongo.MongoClient("f-l2108-pc09.aulas.etsit.urjc.es", port=21000)
    col = con["cstrack"]["geomap_full"]
    info = pd.DataFrame(list(col.find()))
    return info