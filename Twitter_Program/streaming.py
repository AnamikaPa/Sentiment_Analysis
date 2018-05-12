from slistener import SListener
import time, tweepy, sys

CONSUMER_KEY = "zTP2Php66Gkw5vvbzV84poG7B"
CONSUMER_SECRET = "AL8T0XNYA7zrADy6kxL23GUDIVYt9G1qzdayQQ56B1wtRyMMTd"
ACCESS_TOKEN = '4761432372-LhIy3WFhM9w7sNJfhKwyTATF8Fb4KFdVDFYqmV3'
ACCESS_TOKEN_SECRET = '6kYlqfoAWUQsK4tGhvvb7v94QHlbUackjEJpVGpzg6CwE'

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

def main():
    track = ['#iPhoneX','#iPhone10','iPhoneX','iPhone10','iPhone X','iPhone 10']
    #track = ['and','the']

    listen = SListener(api, 'user_data')
    stream = tweepy.Stream(auth, listen)

    print "Streaming started..."

    try: 
        stream.filter(track = track)
    except:
        print "error!"
        stream.disconnect()

if __name__ == '__main__':
    main()