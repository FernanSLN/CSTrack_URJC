import tweepy_credentials as tc
import tweepy
import time
import pickle
import pandas as pd

def limitHandled(cursor):
  while True:
    try:
        yield cursor.next()
    except tweepy.TweepError:
        time.sleep(15 * 60)

auth = tweepy.OAuthHandler(tc.API_KEY, tc.API_SECRET)
auth.set_access_token(tc.ACCESS_TOKEN, tc.ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

topics = ["SDG1", "SDG2", "SDG3", "SDG4", "SDG5", "SDG6", "SDG7", "SDG8", "SDG9", "SDG10", "SDG11",
          "SDG12", "SDG13", "SDG14", "SDG15", "SDG16", "SDG17", "ODS1", "ODS2", "ODS3", "ODS4", "ODS5", "ODS6", "ODS7",
          "ODS8", "ODS9", "ODS10", "ODS11", "ODS12", "ODS13", "ODS14", "ODS15", "ODS16", "ODS17", "goal1", "goal2", "goal3",
          "goal4", "goal5","goal6", "goal7", "goal8", "goal9", "goal10", "goal11", "goal12", "goal13", "goal14", "goal15",
          "goal16", "goal17"]
if __name__ == '__main__':
    tweet_list = []
    for topic in topics:

            tweets = tweepy.Cursor(api.search, q=topic, count=5000, monitor_rate_limit=True,
                                   wait_on_rate_limit=True, wait_on_rate_limit_notify=True,
                                   retry_count=5, retry_delay=5, tweet_mode="extended", result_type="recent").items(5000)
            try:
                for tweet in tweets:
                    #print(page.full_text)
                    print(tweet.id)
                    tweet_list.append({"category": topic, "headline": tweet.full_text, "t_id": tweet.id})
            except Exception as e:
                print(e)
        #except tweepy.error.TweepError:
        #   time.sleep(120)
    df = pd.DataFrame(tweet_list)
    df.to_json("./tweets.json", orient="records")
    df.to_csv("./tweets.csv")