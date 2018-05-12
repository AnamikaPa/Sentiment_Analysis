import twitter, re, datetime, pandas as pd
'''


CONSUMER_KEY = "7RqMuLZrC4MQeEi9BbAR9KRND"
CONSUMER_SECRET = "Eg0BVQOmh3TQoaAhNC44itJjOfrzywNEm8DZsuIQVlNZEolesU"
ACCESS_TOKEN = '866665133380050944-1WPrKPSNTZPCKPQvYZQ4KUzrpqYKw8y'
ACCESS_TOKEN_SECRET = 'KpvUA7ntqWJ966RxlCnn6rquwG9MXmFhWhZLtP8f2ahg7'
'''
class twitterminer():

    request_limit   =   20    
    api             =   False
    data            =   []
   
	
    twitter_keys = {
        'consumer_key': "OCDxryFzCmDnngoMfrhRzxLdF"      , #add your consumer key
        'consumer_secret': "lPtZ1CCuuJLnJSjhTUDajjtxoX0NVgVNZHbjoW5UnlcORmdkzr"   , #add your consumer secret key
        'access_token_key':  "866665133380050944-pRqHcDcBMMupvDfSAifRjnIpcGIGMnX"  , #add your access token key
        'access_token_secret': "zb183wyV8aXUrgBDOMOk4yyUjdGfbOeBYVjkAIVF1jGCL" #add your access token secret key
    }
    
    def __init__(self,  request_limit = 100):
        
        self.request_limit = request_limit
        
        # This sets the twitter API object for use internall within the class
        self.set_api()
        
    def set_api(self):
        
        self.api = twitter.api(
            consumer_key         =   self.twitter_keys['consumer_key'],
            consumer_secret      =   self.twitter_keys['consumer_secret'],
            access_token_key     =   self.twitter_keys['access_token_key'],
			access_token_secret  =   self.twitter_keys['access_token_secret']
        )

    def mine_user_tweets(self, user="anamikap24", mine_retweets=False):

        statuses   =   self.api.GetUserTimeline(screen_name=user, count=self.request_limit)
        data       =   []
		
        for item in statuses:

            mined = {
                        'tweet_id': item.id,
                        'handle': item.user.name,
                        'retweet_count': item.retweet_count,
                        'text': item.text,
                        'mined_at': datetime.datetime.now(),
                        'created_at': item.created_at
                    }
                    
            data.append(mined)
        #status = self.api.PostUpdate('I love python-twitter!')    
        return statuses

    def favourite(self,user):


        status = self.api.GetFollowers(screen_name=user,count = self.request_limit)
        # for item in status:
        #     print(item.users.name)
        print(status)

    


miner = twitterminer()

# insert handle we like
trump_tweets = miner.mine_user_tweets("countermukul")
#trump_df = pd.DataFrame(trump_tweets)
print(trump_tweets)