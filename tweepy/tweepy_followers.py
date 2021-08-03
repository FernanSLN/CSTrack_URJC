import tweepy_credentials as tc
import tweepy
from utils import utils
import networkx as nx
import time
import pandas as pd


def read_followers(filename):
    try:
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        print("Follower file not found")
        return None


def count_times_followed(edges, user):
    sum = 0
    for e in edges:
        if e[1] == user:
            sum = sum + 1
    return sum

def follower_stored(edges, username, follower):
    found = False
    i = 0

    while i < len(edges) and not found:
        if edges[i][1] == username and edges[i][2] == follower:
            found = True
        i += 1
    return found


def get_follower_graph(users):
    g = nx.Graph()
    auth = tweepy.OAuthHandler(tc.API_KEY, tc.API_SECRET)
    auth.set_access_token(tc.ACCESS_TOKEN, tc.ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    edges = []
    df = read_followers("follower_edges.csv")
    df = df[["Source","Target","IdSource"]]
    if df is None:
        df = pd.DataFrame(columns=["Source", "Target", "IdSource"])
    else:
        edges = list(df.to_records(index=False))
        print("EDGES", len(edges))
    for u in users:
        counted_followers = count_times_followed(edges, u)
        n_followers = api.get_user(u).followers_count
        print("USER:", u, counted_followers, n_followers)
        if counted_followers < n_followers / 2:
                pages = tweepy.Cursor(api.followers_ids, screen_name=u).pages()
                while True:
                    try:
                        page = pages.next()
                        print("LEN PAGE FOLLOWERS: ", len(page))
                        for id in page:
                            exists = follower_stored(edges, u, id)
                            if not exists:
                                follower = api.get_user(id)
                                print("Follower: ", follower.screen_name, "User: ", u, "Exists:", exists)
                                edges.append((follower.screen_name, u, id))
                                new_row = {'Source': follower.screen_name, 'Target': u, 'IdSource': id}
                                df = df.append(new_row, ignore_index=True)
                                g.add_edge(follower.screen_name, u)
                        df.to_csv("follower_edges.csv")
                        print(df)
                    except tweepy.error.RateLimitError:
                        print("--- Waiting rate ---")
                        time.sleep(60 * 15)
                    except StopIteration:
                        df.to_csv("follower_edges.csv")
                        break
                    except tweepy.error.TweepError:
                        df.to_csv("follower_edges.csv")
    df.to_csv("follower_edges.csv")
    return g


retweetList = utils.get_retweets("lynguo.csv")
retweetEdges = utils.get_edges(retweetList)
G = nx.DiGraph()
G.add_edges_from(retweetEdges)


f_g = get_follower_graph(G.nodes(data=False))
print("NODES: ", len(f_g.nodes(data=False)))