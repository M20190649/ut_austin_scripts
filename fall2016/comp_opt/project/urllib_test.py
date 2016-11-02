from bs4 import BeautifulSoup
import urllib

post_params = {
    param1 : val1,
    param2 : val2,
    param3 : val3
        }
post_args = urllib.urlencode(post_params)

url = 'https://www.tripadvisor.com/Attractions'
fp = urllib.urlopen(url, post_args)
soup = BeautifulSoup(fp)