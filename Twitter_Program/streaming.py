from slistener import SListener
import time, tweepy, sys

CONSUMER_KEY = ""
CONSUMER_SECRET = ""
ACCESS_TOKEN = ''
ACCESS_TOKEN_SECRET = ''

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
