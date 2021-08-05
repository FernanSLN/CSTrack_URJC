import utils
import networkx as nx
from webweb import Web


def generate_graphs(retweetEdges, mentionEdges):
    ct_rt_graphs = {"retweets": {}, "cites": {}}
    ct_rt_graphs["retweets"]["full"] = nx.Graph()
    ct_rt_graphs["cites"]["full"] = nx.Graph()
    ct_rt_graphs["retweets"]["full"].add_edges_from(retweetEdges)
    ct_rt_graphs["cites"]["full"].add_edges_from(mentionEdges)
    ct_rt_graphs["retweets"]["divided"] = utils.get_subgraphs(ct_rt_graphs["retweets"]["full"])
    ct_rt_graphs["cites"]["divided"] = utils.get_subgraphs(ct_rt_graphs["cites"]["full"])
    return ct_rt_graphs


def display_subgraphs(graphs):
    #we only get graphs with more than 5 nodes
    subgraphs = [graph for graph in graphs if len(graph.nodes) > 5]
    print("Number Nodes:", len(subgraphs[0].nodes))
    web = Web(title="retweets", nx_G=subgraphs[0])
    web.display.gravity = 1
    name = "graph"
    for i in range(2, len(subgraphs)):
        print("Number Nodes:", len(subgraphs[i].nodes))
        web.networks.retweets.add_layer(nx_G=subgraphs[i])

    # show the visualization
    web.show()


retweetList = utils.get_retweets("lynguo_format.csv", ["learn", "educa", "school"])
retweetEdges = utils.get_edges(retweetList)

mentionList = utils.get_cites("lynguo_format.csv")
mentionEdges = utils.get_edges(mentionList)

ct_rt_graphs = generate_graphs(retweetEdges, mentionEdges)

print(len(ct_rt_graphs["retweets"]["full"].nodes))
display_subgraphs(ct_rt_graphs["retweets"]["divided"])