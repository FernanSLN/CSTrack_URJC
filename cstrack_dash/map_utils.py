import plotly.express as px

def get_map_stats_by_country(df, type):
    result = df.groupby(["iso_3", "continent", "country"])[type].sum().reset_index(name=type)
    fig = px.scatter_geo(result, locations="iso_3", width=1500, height=768,
                         color="continent",
                         hover_name="country",
                         size_max = 50,
                         size=type # size of markers, "pop" is one of the columns of gapminder
                         )
    return fig


def get_map_locations(df):
    test = df
    test["country"] = test["country"].fillna(55)
    fig = px.scatter_geo(df, lat="lat", lon="lon", width=1500, height=768,
                         color="country",
                         hover_name = "screen_name"
                         )
    fig.show()