""" Module to create the sidebar of the Dash application
"""

import dash_table
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import style

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
                dbc.Col("Networks"),
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
            dbc.NavLink("Retweets", href="/graph/retweets"),
            dbc.NavLink("RT Communities", href="/graph/retweet_communities"),
            dbc.NavLink("Two mode", href="/graph/two_mode"),
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

submenu_7 = [
    html.Li(
        dbc.Row(
            [
                dbc.Col("Topic modelling"),
                dbc.Col(
                    html.I(className="fas fa-chevron-right mr-3"), width="auto"
                ),
            ],
            className="my-1",
        ),
        style={"cursor": "pointer"},
        id="submenu-7",
    ),
    dbc.Collapse(
        [
            dbc.NavLink("Intertopic map", href="/topic/intertopic"),
            dbc.NavLink("Topic words scores", href="/topic/wordscores"),
        ],
        id="submenu-7-collapse",
    ),
]


