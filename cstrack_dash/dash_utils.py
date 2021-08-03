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
import hashlib
from collections import Counter
from datetime import date
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


def get_graph_figure(G, i=0):
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

def get_hash_name_list(nodes):
    dict_names = {}
    for node in nodes:
        dict_names[node] = hashlib.md5(str(node).encode()).hexdigest()
    return dict_names


def get_community_graph(g, community, i=0):
    c = nx.DiGraph()
    for node in community[i]:
        list_edges = g.edges(node)
        list_edges = [edge for edge in list_edges if edge[1] in community[i]]
        c.add_edges_from(list_edges)
    print(c.nodes)
    dict_names = get_hash_name_list(c.nodes)
    print(dict_names)
    #c = nx.relabel_nodes(c, dict_names)
    return c

def get_communities(g, algorithm="louvain"):
    n_g = nk.nxadapter.nx2nk(g)
    idmap = dict((u, id) for (id, u) in zip(g.nodes(), range(g.number_of_nodes())))
    if algorithm == "louvain":
        communities = nk.community.detectCommunities(n_g)
    else:
        communities = nk.community.detectCommunities(n_g, algo=nk.community.PLP(n_g))
    list_communities = []
    for i in range(0, communities.numberOfSubsets()):
        list_members = []
        for member in communities.getMembers(i):
            list_members.append(idmap[member])
        if len(list_members) > 5:
            list_communities.append(list_members)
    list_communities = [community for community in list_communities if len(community) > 10]
    return list_communities

def kcore_graph(df, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None, anonymize=False):
    df = utils.filter_by_interest(df, interest)
    df = utils.filter_by_topic(df, keywords, stopwords)
    dfRT = df[['Usuario', 'Texto']]
    idx = dfRT['Texto'].str.contains('RT @', na=False)
    dfRT = dfRT[idx]
    rt_edges_list = [list(x) for x in dfRT.to_numpy()]

    edges = []
    for row in rt_edges_list:
        reg = re.search('@(\w+)', row[1])
        if reg:
            matchRT = reg.group(1)
            row[1] = matchRT
            #row[1] = hashlib.md5(matchRT.encode()).hexdigest()
            edges.append(row)

    G = utils.make_weightedDiGraph(edges)
    G.remove_edges_from(nx.selfloop_edges(G))
    core_number = nx.core_number(G)
    values = list(core_number.values())
    degree_count = Counter(values)
    G_kcore = nx.k_core(G, k=2)
    if anonymize:
        dict_labels = get_hash_name_list(G_kcore.nodes)
        G_kcore = nx.relabel_nodes(G_kcore, mapping=dict_labels)
    print(len(G_kcore.nodes))
    """G_kcore_undirected = nx.to_undirected(G_kcore)
    subgraphs = utils.get_subgraphs(G_kcore_undirected)
    subgraphs = [graph for graph in subgraphs if len(graph.nodes) > 5]
    subgraphs = utils.direct_subgraphs(subgraphs)"""

    return G_kcore


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
    fig.update_xaxes(tickangle=90)
    return fig


def get_temporal_figure(df, n_hashtags=5):
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
    for hashtag in elements[:100]:
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
    print(df)
    hashmain = utils.get_hashtagsmain(df, keywords=k, stopwords=stop)
    print("HASHMAIIIN")
    print(hashmain)
    edges = utils.get_edgesMain(hashmain)
    print("EDGEEEEEEEEEEEEEES")
    print(edges)
    # Con las stopwords eliminamos el bot:
    sortedNumberHashtags, sortedHashtagsmain = utils.prepare_hashtagsmain(edges, stopwords=stop)
    df_hashtags = pd.DataFrame(list(zip(sortedHashtagsmain, sortedNumberHashtags)), columns=["Hashtags", "Count"])
    return df_hashtags


def get_all_temporalseries(df, k=None, stop=None):
    df = df[['Usuario', 'Texto', 'Fecha']].copy()

    df = df.dropna()

    df = df[df['Fecha'].str.match('[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]\s[0-9][0-9]:[0-9][0-9]:[0-9][0-9]')]

    df["Fecha"] = pd.to_datetime(df['Fecha'], format="%Y-%m-%d %H:%M:%S").dt.date
    print("DF Tras fechas")
    print(df)
    dias = utils.getDays(df)

    listHt = utils.get_hashtagsmain(df, keywords=k)
    edges = utils.get_edgesMain(listHt)
    sortedNH, sortedMH = utils.prepare_hashtagsmain(edges)
    return df, dias, sortedMH


def get_rt_temporalseries(df, k=None, stop=None):
    df = df[['Usuario', 'Texto', 'Fecha']].copy()

    df = df.dropna()
    df = df[df['Fecha'].str.match('[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]\s[0-9][0-9]:[0-9][0-9]:[0-9][0-9]')]
    df["Fecha"] = pd.to_datetime(df['Fecha'], format="%Y-%m-%d %H:%M:%S").dt.date
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
    db = pymongo.MongoClient(host="f-l2108-pc09.aulas.etsit.urjc.es", port=21000)
    twitter_data = db["cstrack"]["cstrack_stats"]
    twitter_dict_list = list(twitter_data.find())
    df = pd.DataFrame(twitter_dict_list)
    df = df.dropna()
    df["Date"] = pd.to_datetime(df['Date'], format="%d/%m/%Y %H:%M", errors="ignore")
    df["Date"] = pd.to_datetime(df['Date'], format="%d/%m/%Y", errors="ignore")
    return df

def get_two_mode_graph(df, keywords=None):
    return utils.get_twomodeRT(df, keywords)

def get_controls_community2(communities):
    dropdown_options = []
    dropdown_options.append({"label": "all", "value": "all"})
    for i in range(0, len(communities)):
        dropdown_options.append({"label": str(i), "value": i})

    controls = dbc.Form(
        [
            dbc.FormGroup(
                [
                    dbc.Label("Community:"),
                    dcc.Dropdown(
                        id="com_number2",
                        options=dropdown_options,
                        value="all",
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
                        id="com_algorithm2",
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

def get_controls_rt(number_id, keyword_id):
    today = date.today()

    controls = dbc.Form(
        [
            dbc.Row([
                dbc.Col(
                    dbc.FormGroup(
                        [
                            dbc.Label("Number hashtags:"),
                            dbc.Input(id=number_id, style={"width": "100px"}, n_submit=0, min=1,
                                      type="number", value=10, debounce=True),
                        ]
                    ), md=4),
                dbc.Col(
                    dbc.FormGroup(
                        [
                            dbc.Label("Keywords:"),
                            dbc.Input(id=keyword_id, n_submit=0, type="text", value="", debounce=True),
                        ],
                    )),
                dbc.Col(
                    dbc.FormGroup(
                        [
                            dbc.Label("Keywords:"),
                            get_topic_file(keyword_id + "-upload")
                        ],
                    ))
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.FormGroup(
                        [
                            dbc.Label("Dates:"),
                            dcc.DatePickerRange(
                                id='sessions_date',
                                min_date_allowed=(2020, 9, 29),
                                max_date_allowed=(today.year, today.month, today.day),
                                display_format="DD-MM-Y",
                                clearable=True
                            ),
                        ]
                    ),
                ])
            ])
        ],
    )
    return controls

def get_controls_rt_g(keyword_id):
    controls = dbc.Form(
        [
            dbc.FormGroup(
                [
                    dbc.Label("Keywords:"),
                    dbc.Input(id=keyword_id, n_submit=0, type="text", value="", debounce=True),
                ],
                className="mr-3"
            ),
            dbc.FormGroup(
                [
                    dbc.Label("Topics:"),
                    get_topic_file(keyword_id + "-upload")
                ],
            ),
        ],
        inline=True
    )
    return controls

def get_topic_file(id):
    upload_html = dcc.Upload(
        id=id,
        children=html.Div([
            'Upload ',
            html.A('File')
        ]),
        style={
            'width': '100%',
            'height': "calc(1.5em + .75rem + 2px)",
            'lineHeight': '35px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
        },
        # Allow multiple files to be uploaded
    )
    return upload_html

def get_controls_ts(number_id, keyword_id, dc_id, df_ts):
    options = []
    for c in df_ts.columns[:-1]:
        options.append({"label": c, "value": c})

    today = date.today()

    controls = dbc.Form(
        [
            dbc.Row([
                dbc.Col(
                    dbc.FormGroup(
                        [
                            dbc.Label("Number hashtags:"),
                            dbc.Input(id=number_id, style={"width": "100px"}, n_submit=0, min=1, type="number", value=5, debounce=True),
                        ]
                    ), md=4),
                dbc.Col(
                    dbc.FormGroup(
                        [
                            dbc.Label("Keywords:"),
                            dbc.Input(id=keyword_id, n_submit=0, type="text", value="", debounce=True),
                        ],
                    )),
                dbc.Col(
                    dbc.FormGroup(
                        [
                            dbc.Label("Keywords:"),
                            get_topic_file(dc_id + "-upload")
                        ],
                    ))
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.FormGroup(
                        [
                            dbc.Label("Dates:"),
                            dcc.DatePickerRange(
                                id='sessions_date',
                                min_date_allowed=(2020, 9, 29),
                                display_format="DD-MM-Y",
                                max_date_allowed=(today.year, today.month, today.day),
                                clearable=True
                            ),
                        ]
                    ),
                ]),
                dbc.Col(
                    dbc.FormGroup(
                        [
                            dbc.Label("Hashtags:"),
                            dcc.Dropdown(id=dc_id, options=options, multi=True, style={"width": "200px"}),
                        ],
                    ))
            ])
        ]
    )
    return controls




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

def get_controls_topics(number_id, keyword_id, topics):
    today = date.today()

    controls = dbc.Form(
        [
            dbc.Row([
                dbc.Col(
                    dbc.FormGroup(
                        [
                            dbc.Label("Number topics:"),
                            dbc.Input(id=number_id, style={"width": "100px"}, n_submit=0, min=1, max=topics, type="number", value=20, debounce=True),
                        ]
                    ), md=4),
                dbc.Col(
                    dbc.FormGroup(
                        [
                            dbc.Label("Keywords:"),
                            dbc.Input(id=keyword_id, n_submit=0, type="text", value="", debounce=True),
                        ],
                    )),
                dbc.Col(
                    dbc.FormGroup(
                        [
                            dbc.Label("Topics:"),
                            get_topic_file(keyword_id + "-upload")
                        ],
                    ))
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.FormGroup(
                        [
                            dbc.Label("Dates:"),
                            dcc.DatePickerRange(
                                id='sessions_date',
                                min_date_allowed=(2020, 9, 29),
                                display_format="DD-MM-Y",
                                max_date_allowed=(today.year, today.month, today.day),
                                clearable=True
                            ),
                        ]
                    ),
                ])
            ])
        ],
    )
    return controls