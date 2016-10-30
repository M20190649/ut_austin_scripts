from urllib import urlencode
import requests
params = {'q': '7420 Westlake Ter #1210 20817'}
search_url = 'http://www.google.com'
url = search_url + urlencode(params)
r = requests.get(url)