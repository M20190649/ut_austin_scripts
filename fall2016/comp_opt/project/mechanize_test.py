import urllib
import urllib2
import webbrowser
import requests
import lxml.html
import mechanize

url = 'https://www.tripadvisor.com/Attractions'
# values = {'q' : 'austin' }
br = mechanize.Browser()
br.set_handle_robots(False)
br.open(url)
# print [form for form in br.forms()][0] # Tried to see all forms
br.select_form(nr=0)
# print form
# br.select_form(name='q')

br['q'] = 'austin'
response = br.submit()
# for link in br.links(): 
# 	print link.text, link.url
# req = br.click(type='submit',nr=1)
# req = br.submit(label='Search')
# req = br.click_link(label='Search')
# print br.response().read()
# webbrowser.open_new( br.response() )
print response.geturl()
# br.open(req)

# data = urllib.urlencode(values)
# req = urllib2.Request(url, data)
# response = urllib2.urlopen(req)
# print response.read()
# webbrowser.open_new( data )
# response = requests.post(url,data=values)
# print response.content
# tree = lxml.html.document_fromstring(response.content)
# print tree.xpath()

# the_page = response.read()