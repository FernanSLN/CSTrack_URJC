"""
This app creates a simple sidebar layout using inline style arguments and the
dbc.Nav component.
dcc.Location is used to track the current location. There are three callbacks,
one uses the current location to render the appropriate page content, the other
two are used to toggle the collapsing sections in the sidebar. They control the
collapse component and the CSS that rotates the chevron icon respectively.
For more details on building multi-page Dash applications, check out the Dash
documentation: https://dash.plot.ly/urls
"""

#See who are the followers (Projects, people) -> How are we connected, with which other projects, institutions,-> Know about neighbours
#weighted count hashtags
#Exclude and hide in visualizations
#visualization taking into account outdegree
#sentiment analysis - For internal analysis only -> Bot analysis vs person analysis -> Take into account strange phenomena.
#Complex networks conference
import dash
import pandas as pd
import dash_utils
import dash_table
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import generate_utils as gu
from dash.dependencies import Input, Output, State
import style
from generate_utils import filter_by_topic
import map_utils as mu
from webweb import Web
from flask import Flask
from flask_caching import Cache
import communities_utils as cu

# data load
#kcor algorithm
print("DATA LOAD")
com_algorithm = "louvain"
df_map = dash_utils.get_map_df()
df = pd.read_csv("Lynguo_def2.csv", sep=';', encoding='latin-1', error_bad_lines=False)
df_all_h = dash_utils.get_all_hashtags(df)
df_rt_h = dash_utils.get_rt_hashtags(df)
df_ts_raw, days, sortedMH = dash_utils.get_all_temporalseries(df)
df_ts = dash_utils.get_df_ts(df_ts_raw, days, sortedMH)
df_ts_rt_raw, days_rt, sortedMH_rt = dash_utils.get_rt_temporalseries(df)
df_ts_rt = dash_utils.get_df_ts(df_ts_rt_raw, days_rt, sortedMH_rt)
wc_main = dash_utils.wordcloudmain(df)
df_deg = dash_utils.get_degrees(df)
df_sentiment = gu.sentiment_analyser((df))
#df_deg.to_csv("dashdeg.csv")

df_cstrack = dash_utils.get_twitter_info_df()
G = dash_utils.get_graph_rt(df)
communities = dash_utils.get_communities(G)
com = dash_utils.get_community_graph(G,communities)

graph_communities = []
for i in range(0, len(communities)):
    graph_communities.append(dash_utils.get_community_graph(G, communities, i))
print("Termina")
g_communities = cu.get_communities_representative_graph(G, communities)


# link fontawesome to get the chevron icons
FA = "https://use.fontawesome.com/releases/v5.8.1/css/all.css"

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP, FA])
app.config.suppress_callback_exceptions = True
cache = Cache(app.server, config={'CACHE_TYPE': 'simple'})

#server = app.server
print("HELLO")
submenu_1 = [
    html.Li(
        # use Row and Col components to position the chevrons
        dbc.Row(
            [
                dbc.Col("Most used hashtags"),
                dbc.Col(
                    html.I(className="fas fa-chevron-right mr-3"), width="auto"
                ),
            ],
            className="my-1",
        ),
        style={"cursor": "pointer"},
        id="submenu-1",
    ),
    # we use the Collapse component to hide and reveal the navigation links
    dbc.Collapse(
        [
            dbc.NavLink("All hashtags", href="/hashtags/all"),
            dbc.NavLink("Retweeted hashtags", href="/hashtags/rt"),
        ],
        id="submenu-1-collapse",
    ),
]

submenu_2 = [
    html.Li(
        dbc.Row(
            [
                dbc.Col("Time series"),
                dbc.Col(
                    html.I(className="fas fa-chevron-right mr-3"), width="auto"
                ),
            ],
            className="my-1",
        ),
        style={"cursor": "pointer"},
        id="submenu-2",
    ),
    dbc.Collapse(
        [
            dbc.NavLink("All hashtags", href="/timeseries/allhashtags"),
            dbc.NavLink("Retweeted hashtags", href="/timeseries/rthashtags"),
        ],
        id="submenu-2-collapse",
    ),
]

submenu_3 = [
    html.Li(
        dbc.Row(
            [
                dbc.Col("Wordcloud"),
                dbc.Col(
                    html.I(className="fas fa-chevron-right mr-3"), width="auto"
                ),
            ],
            className="my-1",
        ),
        style={"cursor": "pointer"},
        id="submenu-3",
    ),
    dbc.Collapse(
        [
            dbc.NavLink("Wordcloud", href="/wordcloud"),
        ],
        id="submenu-3-collapse",
    ),
]

submenu_4 = [
    html.Li(
        dbc.Row(
            [
                dbc.Col("Tables"),
                dbc.Col(
                    html.I(className="fas fa-chevron-right mr-3"), width="auto"
                ),
            ],
            className="my-1",
        ),
        style={"cursor": "pointer"},
        id="submenu-4",
    ),
    dbc.Collapse(
        [
            dbc.NavLink("Degrees", href="/tables/retweets"),
            dbc.NavLink("Sentiment", href="/tables/sentiment"),
        ],
        id="submenu-4-collapse",
    ),
]

submenu_5 = [
    html.Li(
        dbc.Row(
            [
                dbc.Col("Communities"),
                dbc.Col(
                    html.I(className="fas fa-chevron-right mr-3"), width="auto"
                ),
            ],
            className="my-1",
        ),
        style={"cursor": "pointer"},
        id="submenu-5",
    ),
    dbc.Collapse(
        [
            dbc.NavLink("Retweet", href="/graph/retweet_communities"),
            dbc.NavLink("Hashtag", href="/graph/hashtag_communities"),
        ],
        id="submenu-5-collapse",
    ),
]


submenu_6 = [
    html.Li(
        dbc.Row(
            [
                dbc.Col("Geomaps"),
                dbc.Col(
                    html.I(className="fas fa-chevron-right mr-3"), width="auto"
                ),
            ],
            className="my-1",
        ),
        style={"cursor": "pointer"},
        id="submenu-6",
    ),
    dbc.Collapse(
        [
            dbc.NavLink("Tweets and Follows per country", href="/geomap/activity"),
            dbc.NavLink("Locations", href="/geomap/locations"),
        ],
        id="submenu-6-collapse",
    ),
]

sidebar = html.Div(
    [
        dbc.Col(html.Img(src=app.get_asset_url('cstrack_logo.png'), height="50px")),
        html.Hr(),
        html.P(
            "Available graphs", className="lead"
        ),
        dbc.Nav([dbc.NavLink("CS-Track stats", href="/", active="exact")] + submenu_1 + submenu_2 + submenu_3 + submenu_4 + submenu_5 + submenu_6, vertical=True),
    ],
    style=style.SIDEBAR_STYLE,
    id="sidebar",
)

content = html.Div(id="page-content", style=style.CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


# this function is used to toggle the is_open property of each Collapse
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


# this function applies the "open" class to rotate the chevron
def set_navitem_class(is_open):
    if is_open:
        return "open"
    return ""


for i in range(1, 7):
    app.callback(
        Output(f"submenu-{i}-collapse", "is_open"),
        [Input(f"submenu-{i}", "n_clicks")],
        [State(f"submenu-{i}-collapse", "is_open")],
    )(toggle_collapse)

    app.callback(
        Output(f"submenu-{i}", "className"),
        [Input(f"submenu-{i}-collapse", "is_open")],
    )(set_navitem_class)


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        controls = dash_utils.get_controls_rt("input-rt-cstrack", "hashtag-rt-cstrack")
        html_plot = html.Div(children=[
            dcc.Loading(
                # style={"height":"200px","font-size":"100px","margin-top":"500px", "z-index":"1000000"},
                style=style.SPINER_STYLE,
                color="#000000",
                id="loading-1",
                type="default",
                children=html.Div(id="loading-output", children=[
                    dbc.Row(controls, justify="center"),
                    dbc.Row(
                        dbc.Col(dcc.Graph(id='graph_retweets', figure=dash_utils.get_cstrack_graph(df_cstrack, "Retweets", "Retweets received")), md=8),
                        justify="center"),
                    dbc.Row(
                        dbc.Col(dcc.Graph(id='graph_followers', figure=dash_utils.get_cstrack_graph(df_cstrack, "Followers", "Followers")),
                                md=8),
                        justify="center"),
                    dbc.Row(
                        dbc.Col(dcc.Graph(id='graph_followers',
                                          figure=dash_utils.get_cstrack_graph(df_cstrack, "Tweets", "Number of Tweets")),
                                md=8),
                        justify="center")
                ])
            ),
        ]),
        return html_plot
    elif pathname == "/hashtags/all":
        print("PATH 1")
        controls = dash_utils.get_controls_rt("input-key-all", "hashtag-number-all")
        """graph_fig = dbc.Col(dcc.Graph(id='graph_all_hashtags', figure=dash_utils.get_figure(df_all_h[0:10])), md=8)
        graph = dash_utils.set_loading(controls, graph_fig)"""
        html_plot = html.Div(children=[
            dcc.Loading(
                # style={"height":"200px","font-size":"100px","margin-top":"500px", "z-index":"1000000"},
                style=style.SPINER_STYLE,
                color="#000000",
                id="loading-1",
                type="default",
                children=html.Div(id="loading-output", children=[
                    dbc.Row(controls, justify="center"),
                    dbc.Row(
                        dbc.Col(dcc.Graph(id='graph_all_hashtags', figure=dash_utils.get_figure(df_all_h[0:10])), md=8),
                        justify="center")
                ])
            ),
        ]),

        return html_plot
    elif pathname == "/hashtags/rt":
        controls = dash_utils.get_controls_rt("input-key-rt", "hashtag-number-rt")
        html_plot = html.Div(children=[
            dcc.Loading(
                # style={"height":"200px","font-size":"100px","margin-top":"500px", "z-index":"1000000"},
                style=style.SPINER_STYLE,
                color="#000000",
                id="loading-1",
                type="default",
                children=html.Div(id="loading-output", children=[
                    dbc.Row(controls, justify="center"),
                    dbc.Row(
                        dbc.Col(dcc.Graph(id='graph_rt_hashtags', figure=dash_utils.get_figure(df_rt_h[0:10])), md=8),
                        justify="center")
                ])
            ),

        ]),
        return html_plot
    elif pathname == "/timeseries/allhashtags":
        print("PATH 2")
        controls = dash_utils.get_controls_ts("input-key-ts-all", "hashtag-number-ts-all", "hashtags-name-ts-all", df_ts)
        html_plot = html.Div(children=[
            dcc.Loading(
                # style={"height":"200px","font-size":"100px","margin-top":"500px", "z-index":"1000000"},
                style=style.SPINER_STYLE,
                color="#000000",
                id="loading-1",
                type="default",
                children=html.Div(id="loading-output", children=[
                    dbc.Row(controls, justify="center"),
                    dbc.Row(dbc.Col(dcc.Graph(id='graph_ts_all_hashtags', figure=dash_utils.get_temporal_figure(df_ts)),
                                    md=8), justify="center"),
                ])
            ),
        ]),
        return html_plot

    elif pathname == "/wordcloud":
        controls = dash_utils.get_controls_rt("input-key-ws", "hashtag-number-ts-all")
        html_plot = html.Div(children=[
            dcc.Loading(
                # style={"height":"200px","font-size":"100px","margin-top":"500px", "z-index":"1000000"},
                style=style.SPINER_STYLE,
                color="#000000",
                id="loading-1",
                type="default",
                children=html.Div(id="loading-output", children=[
                    dbc.Row(controls, justify="center"),
                    dbc.Row(dbc.Col(html.Img(src=app.get_asset_url('wc2.png'))), justify="center"),
                ])
            ),
        ]),
        return html_plot
    elif pathname == "/timeseries/rthashtags":
        print("PATH 2")
        controls = dash_utils.get_controls_ts("input-key-ts-rt", "hashtag-number-ts-rt")
        html_plot = html.Div(children=[
            dcc.Loading(
                # style={"height":"200px","font-size":"100px","margin-top":"500px", "z-index":"1000000"},
                style=style.SPINER_STYLE,
                color="#000000",
                id="loading-1",
                type="default",
                children=html.Div(id="loading-output", children=[
                    dbc.Row(controls, justify="center"),
                    dbc.Row(
                        dbc.Col(dcc.Graph(id='graph_rt_all_hashtags', figure=dash_utils.get_temporal_figure(df_ts_rt)),
                                md=8), justify="center"),
                ])
            ),
        ]),
        return html_plot

    elif pathname == "/tables/retweets":
        print("PATH 2")
        html_plot = html.Div(children=[
            dcc.Loading(
                # style={"height":"200px","font-size":"100px","margin-top":"500px", "z-index":"1000000"},
                style=style.SPINER_STYLE,
                color="#000000",
                id="loading-1",
                type="default",
                children=html.Div(id="loading-output", children=[
                    dbc.Row(dbc.Col(

                        dash_table.DataTable(
                            id='datatable-interactivity',
                            columns=[
                                {"name": i, "id": i, "deletable": True, "selectable": True, "hideable": True}
                                if i == "iso_alpha3" or i == "year" or i == "id"
                                else {"name": i, "id": i, "deletable": True, "selectable": True}
                                for i in df_deg.columns
                            ],
                            data=df_deg.to_dict('records'),  # the contents of the table
                            editable=True,  # allow editing of data inside all cells
                            filter_action="native",  # allow filtering of data by user ('native') or not ('none')
                            sort_action="native",  # enables data to be sorted per-column by user or not ('none')
                            sort_mode="single",  # sort across 'multi' or 'single' columns
                            column_selectable="multi",  # allow users to select 'multi' or 'single' columns
                            row_selectable="multi",  # allow users to select 'multi' or 'single' rows
                            row_deletable=True,  # choose if user can delete a row (True) or not (False)
                            selected_columns=[],  # ids of columns that user selects
                            selected_rows=[],  # indices of rows that user selects
                            page_action="native",  # all data is passed to the table up-front or not ('none')
                            page_current=0,  # page number that user is on
                            page_size=10,  # number of rows visible per page
                            style_data={  # overflow cells' content into multiple lines
                                'whiteSpace': 'normal',
                                'height': 'auto'
                            }
                        )
                        ,md=8), justify="center"),
                ])
            ),
        ]),
        return html_plot
    elif pathname == "/tables/sentiment":
        html_plot = html.Div(children=[
            dcc.Loading(
                # style={"height":"200px","font-size":"100px","margin-top":"500px", "z-index":"1000000"},
                style=style.SPINER_STYLE,
                color="#000000",
                id="loading-1",
                type="default",
                children=html.Div(id="loading-output", children=[
                    dbc.Row(dbc.Col(

                        dash_table.DataTable(
                            id='datatable-sentiment',
                            style_cell={"minWidth": "80px", "maxWidth": "500px"},
                            columns=[
                                {"name": i, "id": i, "deletable": True, "selectable": True, "hideable": True}
                                if i == "iso_alpha3" or i == "year" or i == "id"
                                else {"name": i, "id": i, "deletable": True, "selectable": True}
                                for i in df_sentiment.columns
                            ],
                            data=df_sentiment.to_dict('records'),  # the contents of the table
                            editable=True,  # allow editing of data inside all cells
                            filter_action="native",  # allow filtering of data by user ('native') or not ('none')
                            sort_action="native",  # enables data to be sorted per-column by user or not ('none')
                            sort_mode="single",  # sort across 'multi' or 'single' columns
                            column_selectable="multi",  # allow users to select 'multi' or 'single' columns
                            row_selectable="multi",  # allow users to select 'multi' or 'single' rows
                            row_deletable=True,  # choose if user can delete a row (True) or not (False)
                            selected_columns=[],  # ids of columns that user selects
                            selected_rows=[],  # indices of rows that user selects
                            page_action="native",  # all data is passed to the table up-front or not ('none')
                            page_current=0,  # page number that user is on
                            page_size=10,  # number of rows visible per page
                            style_data={  # overflow cells' content into multiple lines
                                'whiteSpace': 'normal',
                                'height': 'auto'
                            }
                        )
                        ,md=8), justify="center"),
                ])
            ),
        ]),
        return html_plot
    elif pathname == "/graph/retweet_communities":
        web = Web(nx_G=g_communities)
        web.display.height = 600
        web.display.gravity = 0.5
        web.save("./assets/test.html")
        srcDoc = open("./assets/test.html").read()
        options = dash_utils.get_controls_community2(communities)
        html_plot = html.Div(children=[
            dcc.Loading(
                # style={"height":"200px","font-size":"100px","margin-top":"500px", "z-index":"1000000"},
                style=style.SPINER_STYLE,
                color="#000000",
                id="loading-1",
                type="default",
                children=html.Div(id="loading-output", children=[
                    dbc.Row(options, justify="center"),
                    dbc.Row(
                        dbc.Col(html.Iframe(id="graph_communities_web", srcDoc=srcDoc, height=800, width=1600), md=8)
                    )
                ])
            ),

        ]),
        return html_plot
    elif pathname == "/geomap/activity":
        print("geomap")
        options = dash_utils.get_controls_activity()
        html_plot = html.Div(children=[
            dcc.Loading(
                # style={"height":"200px","font-size":"100px","margin-top":"500px", "z-index":"1000000"},
                style=style.SPINER_STYLE,
                color="#000000",
                id="loading-1",
                type="default",
                children=html.Div(id="loading-output", children=[
                    dbc.Row(options, justify="center"),
                    dbc.Row(
                        dbc.Col(dcc.Graph(id='geograph', figure=mu.get_map_stats_by_country(df_map)), md=8))
                ])
            ),

        ]),
        return html_plot

    elif pathname =="/geomap/locations":
        html_plot = html.Div(children=[
            dcc.Loading(
                # style={"height":"200px","font-size":"100px","margin-top":"500px", "z-index":"1000000"},
                style=style.SPINER_STYLE,
                color="#000000",
                id="loading-1",
                type="default",
                children=html.Div(id="loading-output", children=[
                    dbc.Row(
                        dbc.Col(dcc.Graph(id='geomap_locations', figure=mu.get_map_locations(df_map)), md=8))
                ])
            ),

        ]),
        return html_plot


    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
        ]
    )


@app.callback(
    Output('graph_all_hashtags', 'figure'),
    [Input("input-key-all", "n_submit"), Input("hashtag-number-all", "n_submit")],
    [State('input-key-all', "value"), State('hashtag-number-all', "value")]
)
def update_hashtags_plot_all(n_submits, n_submits2, hashtag_number, input_key):
    if (n_submits + n_submits2) == 0:
        return dash.no_update
    df_rt = filter_by_topic(df, keywords=input_key.split(","), stopwords=["machinelearning", " ai "])
    df_rt = dash_utils.get_rt_hashtags(df_rt)
    df_rt = df_rt[:hashtag_number]
    return dash_utils.get_figure(df_rt)


@app.callback(
    Output('graph_rt_hashtags', 'figure'),
    [Input("input-key-rt", "n_submit"), Input("hashtag-number-rt", "n_submit")],
    [State('input-key-rt', "value"), State('hashtag-number-rt', "value")]
)
def update_hashtags_plot(n_submits, n_submits2, hashtag_number, input_key):
    if (n_submits + n_submits2) == 0:
        return dash.no_update
    df_r = filter_by_topic(df, keywords=input_key.split(","), stopwords=None)
    df_r = dash_utils.get_rt_hashtags(df_r)
    df_r = df_r[:hashtag_number]
    return dash_utils.get_figure(df_r)


@app.callback(
    Output('graph_ts_all_hashtags', 'figure'),
    [Input("input-key-ts-all", "n_submit"), Input("hashtag-number-ts-all", "n_submit"), Input("hashtags-name-ts-all", "value")],
    [State('input-key-ts-all', "value"), State('hashtag-number-ts-all', "value")]
)
def update_ts_all_plot(n_submits, n_submits2, value_dd, hashtag_number, input_key):
    global df_ts
    print("Number", n_submits, "Submit2", n_submits2, "numbers", hashtag_number, "another", input_key)
    print("VALUE DD", value_dd)
    if (n_submits + n_submits2) == 0 and not value_dd:
        print("NO UPDATE")
        return dash.no_update
    print("PASA POR AQUI")
    if len(input_key) > 0:
        df_ts_raw, days, sortedMH = dash_utils.get_all_temporalseries(df, k=input_key.split(","))
        df_ts = dash_utils.get_df_ts(df_ts_raw, days, sortedMH)

    print("LLEGA AQUI")
    if value_dd and len(value_dd) > 0:
        print("LEN MNAYOOOR")
        print(df_ts)
        df_ts_filtered = df_ts[value_dd + ["date"]]
        print(df_ts_filtered)
        return dash_utils.get_temporal_figure(df_ts_filtered, n_hashtags=hashtag_number)
    return dash_utils.get_temporal_figure(df_ts, n_hashtags=hashtag_number)


@app.callback(
    Output('graph_ts_rt_hashtags', 'figure'),
    [Input("input-key-ts-rt", "n_submit"), Input("hashtag-number-ts-rt", "n_submit")],
    [State('input-key-ts-rt', "value"), State('hashtag-number-ts-rt', "value")]
)
def update_ts_rt_plot(n_submits, n_submits2, hashtag_number, input_key):
    print("Number", n_submits, "Submit2", n_submits2, "numbers", hashtag_number, "another", input_key)
    if (n_submits + n_submits2) == 0:
        return dash.no_update
    df_ts_rt_raw, days_rt, sortedMH_rt = dash_utils.get_rt_temporalseries(df)
    df_ts_rt = dash_utils.get_df_ts(df_ts_rt_raw, days_rt, sortedMH_rt)
    return dash_utils.get_temporal_figure(df_ts_rt, n_hashtags=hashtag_number)


@app.callback(
    dash.dependencies.Output('geograph', 'figure'),
    [dash.dependencies.Input('activity_type', 'value')])
def update_com_graph(value):
    fig = mu.get_map_stats_by_country(df_map, value)
    return fig


@app.callback(
    dash.dependencies.Output('graph_communities', 'figure'),
    [dash.dependencies.Input('com_number', 'value'),
     dash.dependencies.Input('com_algorithm', 'value')])
def update_com_graph(value, algorithm):
    global com_algorithm
    global communities
    print(value)
    print(algorithm)
    if com_algorithm != algorithm:
        com_algorithm = algorithm
        communities = dash_utils.get_communities(G, algorithm)
    com = dash_utils.get_community_graph(G,communities, int(value))
    web = Web(nx_G=com)
    web.display.height = 600
    web.save("./assets/test.html")
    srcDoc = open("./assets/test.html").read()
    return dash_utils.get_graph_figure(com, int(value))

@app.callback(
    dash.dependencies.Output('graph_communities_web', 'srcDoc'),
    [dash.dependencies.Input('com_number2', 'value'),
     dash.dependencies.Input('com_algorithm2', 'value')])
def update_com_graph(value, algorithm):
    global com_algorithm
    global communities
    print(value)
    print(algorithm)
    if value == "all":
        com = g_communities
    else:
        if com_algorithm != algorithm:
            com_algorithm = algorithm
            communities = dash_utils.get_communities(G, algorithm)
        com = dash_utils.get_community_graph(G,communities, int(value))
    web = Web(nx_G=com)
    web.display.height = 600
    web.display.gravity = 0.5
    web.save("./assets/test.html")
    srcDoc = open("./assets/test.html").read()
    return srcDoc

if __name__ == "__main__":
    print("HI")
    app.run_server(host="0.0.0.0",port=6123, debug=False)

