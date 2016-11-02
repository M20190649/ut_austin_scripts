import requests
import webbrowser
 
url = 'https://www.tripadvisor.com/Attractions'
payload = {'q':'austin'}
r = requests.get(url, params=payload)
webbrowser.open_new( r )

