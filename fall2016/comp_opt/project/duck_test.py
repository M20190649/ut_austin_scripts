import requests

url = "http://google.com/"
payload = {'q':'python'}
r = requests.post(url, payload)
print r.content