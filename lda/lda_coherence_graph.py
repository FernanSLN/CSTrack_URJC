import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

df_results = pd.read_csv("coherences.csv")
print(df_results)
color_list = []

layout = dict(
    template="simple_white",
    xaxis=dict(ticks="outside", showline=True),
    yaxis=dict(ticks="outside", showline=True),
)


data = go.Scatter(x=df_results["ntopics"].values.tolist(), y=df_results["coherence"].values.tolist(), mode="lines+markers", marker=dict(color="lightsteelblue"))
fig = go.Figure(data, layout)
fig.update_layout({
    "plot_bgcolor" : 'rgba(0,0,0,0)',
})
fig.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickfont = dict(size=20),
        tickvals =[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    ),
    yaxis = dict(
        tickfont = dict(size=20)
    )
)
fig.show()