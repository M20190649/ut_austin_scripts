import urllib2
import urllib
import json

url = "https://developers.google.com/custom-search/"

query = "Austin Torchy's Tacos"

query = urllib.urlencode( {'q' : query } )

response = urllib2.urlopen (url + query ).read()

data = json.loads ( response )

print data

results = data [ 'responseData' ] [ 'results' ]

for result in results:
    title = result['title']
    url = result['url']
    print ( title + '; ' + url )