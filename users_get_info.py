import tweepy
import tweepy_credentials as tc
import pymongo
import pandas as pd

def get_users(api, user_col):
    df = pd.read_csv("Lynguo_def2.csv", sep=';', encoding='latin-1', error_bad_lines=False)
    user_list = df["Usuario"].tolist()
    for user in user_list:
        print(user)
        exists = user_col.find_one({"screen_name": user})
        if exists is None:
            try:
                user = api.get_user(user)
                user_col.insert_one(user._json)
            except tweepy.error.TweepError:
                print(user, "not found")


db = pymongo.MongoClient(host="f-l2108-pc09.aulas.etsit.urjc.es", port=21000)
db = db["cstrack"]
col = db["tweets_cstrack"]
user_col = db["users"]

auth = tweepy.OAuthHandler(tc.API_KEY, tc.API_SECRET)
auth.set_access_token(tc.ACCESS_TOKEN, tc.ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
get_users(api, user_col)