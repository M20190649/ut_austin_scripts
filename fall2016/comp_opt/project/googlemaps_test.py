import googlemaps
from datetime import datetime

gmaps = googlemaps.Client(key='AIzaSyDWv1t2Ry4_03O3-eTEK24wfaoYMEKE240')
address = '1600 Amphitheatre Parkway, Mountain View, CA 94043, USA'
# lat, lon = gmaps.address_to_latlng(address)
# print lat, lon
result = gmaps.geocode(address=address)
print result