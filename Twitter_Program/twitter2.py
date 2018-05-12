from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
#import MySQLdb
import time
import json



#        replace mysql.server with "localhost" if you are running via your own server!
#                        server       MySQL username	MySQL pass  Database name.
#conn = MySQLdb.connect("mysql.server","beginneraccount","cookies","beginneraccount$tutorial")

#c = conn.cursor()

#consumer key, consumer secret, access token, access secret.
ckey="NLE1yzWkOwk5fTWfRSO2Vi8PN"
csecret="BZ2Ciol4Q40kiCP6X5tTfKmMvRSDSf95TCCYN6pszVoYz7UML3"
atoken="1157716766-HmcvLmZK1e5pfFis21OwTprjK1Co90VhRDYXUtb"
asecret="JvpCNGwnfde0xGRyAetqjcQpngR67AyrVTjaCkTMhsYwY"


class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)
        str = json.dumps(all_data,indent=4)
        
        print(str)
        saveFile = open('twit.json','a')
        saveFile.write(str)
        saveFile.write('\n')
        saveFile.close()
        return True

    def on_error(self, status):
        print (status)  

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["car"])
