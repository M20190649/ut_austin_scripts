import geocoder

g = geocoder.google('blanton museum of art').latlng
print g

if len(g) == 0:
	print 'yep'