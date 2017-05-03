import googlemaps
from datetime import datetime
import urllib, json
# pip install Image
import PIL.Image as img

# googleplaces_webservice_key = 'AIzaSyD1iWyK2FNXlRdhU5JKgt2ZRtVYGGWc8-k'		# sudesshc gmail account

googleplaces_webservice_key = 'AIzaSyBsILob4SDyR5z0IurzRUylOjioqiTxXP4'		# sdsh11 gmail account

# # This key is reserved for presentation day; DO NOT USE on any other day
# googleplaces_webservice_key = 'AIzaSyBQAXQgerdCsk-W1r5NimKccumd32lPzPs'		# utexas account

def build_url_text_search(query, output = 'json', key = googleplaces_webservice_key, **kwargs):
	"""
		Builds URL to search for places on Google Maps using Google APIs.
		Text Search returns info about a set of places based on a string: for example, 'pizza in Austin', 'Indian restaurants in Dallas'
		Search Request Form: https://maps.googleapis.com/maps/api/place/textsearch/output?parameters (output: json, xml)
		This method assumes the JSON output.

		Required params:
		key -- API key
		query -- text string on which to search; this param becomes optional when the type param is also used in the search request

		Some Optional params:
		location -- specified as latitude,longitude (NO SPACE after comma); must also specify radius if this param is used
		radius -- distance (in meters); max allowed is 50000; radius is not included if rankby=distance
		language -- en
		minprice and maxprice -- values range from 0 (most affordable) to 4 (most expensive)
		opennow -- Returns only those places that are open for business at the time the query is sent.
		Note: Places that do not specify opening hours in the Google Places database will not be returned if you include this parameter in your query.
		pagetoken -- next 20 results from previously run search
		type -- at most one type allowed (example: accounting; hospital; library; airport, etc.)
		See https://developers.google.com/places/web-service/supported_types for more types
		You may bias results to a specified circle by passing a location and a radius parameter.
		This will instruct the Google Places service to prefer showing results within that circle. Results outside the defined area may still be displayed.

		The below example shows a search for restaurants near Austin.
		https://maps.googleapis.com/maps/api/place/textsearch/xml?query=restaurants+in+Austin&key=YOUR_API_KEY

		Returns a URL.
	"""
	base_url = 'https://maps.googleapis.com/maps/api/place/textsearch/'
	url = base_url + output + '?key=' + key + '&query=' + query
	for key in kwargs:
		url = url + '&{0}={1}'.format(key, kwargs[key])
	return url

def GetResults(myresponse):
	"""
		Returns a dictionary of the relevant info from Text Search.

		myresponse -- JSON object returned by GetResponse()

		Returns a dict with the following keys: place_id, location, viewport, name, address
	"""
	results = []
	if myresponse['status'] == 'OK':
		# print 'Attributions:\n{}'.format(myresponse['html_attributions'])
		count = 1
		for result in myresponse['results']:
			# print '\n\nResult {}'.format(count)
			# if 'icon' in result:
			# 	print 'Icon url: {}'.format(result['icon'])
			# else:
			# 	print 'Icon url: {}'.format('MISSING DATA!')
			# print 'PLace ID: {}'.format(result['place_id'])
			results.append({'place_id': result['place_id']})
			if 'geometry' in result:
				if 'location' in result['geometry']:
					# print 'Geometry location: {}'.format(result['geometry']['location'])
					results[count - 1]['location'] = result['geometry']['location']
				if 'viewport' in result['geometry']:
					# print 'Geometry viewport: {}'.format(result['geometry']['viewport'])
					results[count - 1]['viewport'] = result['geometry']['viewport']
			if 'name' in result:
				# print 'Name: {}'.format(result['name'].encode('utf-8'))
				results[count - 1]['name'] = result['name'].encode('utf-8')
			# else:
			# 	print 'Name: {}'.format('NAME MISSING!')
			# if 'opening_hours' in result:
			# 	print 'Opening Hours: {}'.format(result['opening_hours'])
			# # print 'Photos: {}'.format(result['photos'])
			# if 'scope' in result:
			# 	print 'Scope: {}'.format(result['scope'])	#can be either APP or GOOGLE
			# if 'price_level' in result:
			# 	print 'Price Level: {}'.format(result['price_level'])	# Free (0), Inexpensive, Moderate, Expensive, Very Expensive (4)
			# if 'rating' in result:
			# 	print 'Rating: {}'.format(result['rating'])		
			# if 'types' in result:
			# 	print 'Types: {}'.format(result['types'])
			# if 'vicinity' in result:
			# 	print 'Vicinity: {}'.format(result['vicinity'])
			if 'formatted_address' in result:
				# print 'Address: {}'.format(result['formatted_address'].encode('utf-8'))
				results[count - 1]['address'] = result['formatted_address'].encode('utf-8')
			# if 'permanently_closed' in result:
			# 	print 'Is this place permanently closed? {}'.format(result['permanently_closed'])
			count += 1
	else:
		if myresponse['status'] == 'ZERO_RESULTS':
			print 'No Results found!'
		if myresponse['status'] == 'OVER_QUERY_LIMIT':
			print 'Your over your quota of usage limit!'
		if myresponse['status'] == 'REQUEST_DENIED ':
			print 'Your request has been denied! Please verify your API key.'
		if myresponse['status'] == 'INVALID_REQUEST ':
			print 'Some required parameter is PROBABLY missing!'

	return results

def build_url_place_details(placeid, output = 'json', key = googleplaces_webservice_key, **kwargs):
	"""
		Builds URL to get the details of a place on Google Maps using Google APIs.
		Place details returns info about a particular establishment or point of interest.
		Need a place_id for this request.
		Search Request Form: https://maps.googleapis.com/maps/api/place/details/output?parameters (output: json, xml)
		This method assumes the JSON output.

		Required params:
		key -- API key
		placeid -- textual identifier that uniquely identifies a place

		Some Optional params:
		language -- en

		The following example requests the details of a place by placeid:
		https://maps.googleapis.com/maps/api/place/details/json?placeid=ChIJN1t_tDeuEmsRUsoyG83frY4&key=YOUR_API_KEY

		Returns a URL.
	"""
	base_url = 'https://maps.googleapis.com/maps/api/place/details/'
	url = base_url + output + '?key=' + key + '&placeid=' + placeid
	for key in kwargs:
		url = url + '&{0}={1}'.format(key, kwargs[key])
	return url

def GetPlaceDetails(myresponse, review_count = 3):
	"""
	"""
	output = {}
	if myresponse['status'] == 'OK':
		# if len(myresponse['html_attributions']) > 0:
		# 	print 'Attributions:\n{}'.format(myresponse['html_attributions'])
		# print '\n\nRESULT:\n'
		result = myresponse['result']
		# if 'address_components' in result:
		# 	print 'Address components: {}'.format(result['address_components'])
		if 'formatted_address' in result:
			output['address'] = result['formatted_address'].encode('utf-8')
			# print 'Address: {}'.format(result['formatted_address'].encode('utf-8'))
		else:
			output['address'] = None
			# print 'Address: {}'.format('ADDRESS MISSING')
		# if 'formatted_phone_number' in result:
		# 	print 'Phone Number: {}'.format(result['formatted_phone_number'])
		if 'international_phone_number' in result:
			output['contact'] = result['international_phone_number']
			# print 'Phone Number (with international code): {}'.format(result['international_phone_number'])
		if 'geometry' in result:
			if 'location' in result['geometry']:
				output['location'] = result['geometry']['location']
				# print 'location: ({0}, {1})'.format(result['geometry']['location']['lat'], result['geometry']['location']['lng'])
			if 'viewport' in result['geometry']:
				output['viewport'] = result['geometry']['viewport']
				# print 'Geometry viewport: {}'.format(result['geometry']['viewport'])
		# if 'icon' in result:
		# 	print 'Icon url: {}'.format(result['icon'])
			# urllib.urlretrieve(result['icon'], "icon.png")
			# image = img.open('icon.png').show()
		# else:
		# 	print 'Icon url: {}'.format('MISSING DATA!')
		if 'name' in result:
			output['name'] = result['name'].encode('utf-8')
			# print 'Name: {}'.format(result['name'].encode('utf-8'))
		else:
			output['name'] = None
			# print 'Name: {}'.format('NAME MISSING!')
		if 'opening_hours' in result:
			# print 'Opening Hours:'
			if 'open_now' in result['opening_hours']:
				output['open_now'] = result['opening_hours']['open_now']
				# print '\tOpen now? {}'.format('Yes' if result['opening_hours']['open_now'] == True else 'No')
			# if 'periods' in result['opening_hours']:
			# 	print '\tPeriods: {}'.format(result['opening_hours']['periods'])
			if 'weekday_text' in result['opening_hours']:
				output['weekday_text'] = 'Timings:\n'
				# print '\tTimings:'
				for weekday in result['opening_hours']['weekday_text']:
					output['weekday_text'] = output['weekday_text'] + '\t\t\t\t\t\t\t{}\n'.format(weekday.encode('utf-8'))
					# print '\t\t\t{}'.format(weekday.encode('utf-8'))
		if 'permanently_closed' in result:
			output['perm_closed'] = True
			# print 'Place is PERMANENTLY CLOSED!'
		# print 'Photos: {}'.format(result['photos'])
		if 'place_id' in result:
			output['placeid'] = result['place_id']
			# print 'Place ID: {}'.format(result['place_id'])
		# if 'scope' in result:
		# 	print 'Scope: {}'.format(result['scope'])	#can be either APP or GOOGLE
		if 'price_level' in result:
			# Free (0), Inexpensive, Moderate, Expensive, Very Expensive (4)
			output['price_level'] = {0: 'Free', 1: 'Inexpensive $', 2: 'Moderate $$', 3: 'Expensive $$$', 4: 'Very Expensive $$$$'}.get(result['price_level'], 'Info not available!')
			# print 'Price Level: {}'.format({0: 'Free', 1: 'Inexpensive $', 2: 'Moderate $$', 3: 'Expensive $$$', 4: 'Very Expensive $$$$'}.get(result['price_level'], 'Info not available!'))	
		if 'rating' in result:
			output['rating'] = result['rating']
			# print 'Rating: {}'.format(result['rating'])
		output['reviews'] = 'Reviews:\n'
		# print 'Reviews:'
		if 'reviews' in result:
			if len(result['reviews']) < review_count:
				output['reviews'] = output['reviews'] + '\tOnly {} review(s) available!\n'.format(len(result['reviews']))
				# print '\tOnly {} review(s) available!'.format(len(result['reviews']))
			for i in xrange(0, min(review_count, len(result['reviews']))):
				output['reviews'] = output['reviews'] + '\tReview {0}:\n'.format(i+1, result['reviews'][i])
				# print '\tReview {0}:'.format(i+1, result['reviews'][i])
				if 'author_url' in result['reviews'][i]:
					output['reviews'] = output['reviews'] + '\t\t\tAuthor: {0} ({1})\n'.format(result['reviews'][i]['author_name'].encode('utf-8'), result['reviews'][i]['author_url'])
					# print '\t\t\tAuthor: {0} ({1})'.format(result['reviews'][i]['author_name'].encode('utf-8'), result['reviews'][i]['author_url'])
				output['reviews'] = output['reviews'] + '\t\t\tOverall Rating: {}\n'.format(result['reviews'][i]['rating'])
				# print '\t\t\tOverall Rating: {}'.format(result['reviews'][i]['rating'])
				# aspects: appeal, atmosphere, decor, facilities, food, overall, quality and service; rating: 0 to 3
				output['reviews'] = output['reviews'] + '\t\t\tDetailed Rating (scale: 0 to 3):\n'.format(result['reviews'][i]['aspects'])
				# print '\t\t\tDetailed Rating (scale: 0 to 3):'.format(result['reviews'][i]['aspects'])
				for aspect in result['reviews'][i]['aspects']:
					output['reviews'] = output['reviews'] + '\t\t\t\t\t\t\t\t\t\t\t{0}: {1}\n'.format(aspect['type'], aspect['rating'])
					# print '\t\t\t\t\t\t\t{0}: {1}'.format(aspect['type'], aspect['rating'])
				output['reviews'] = output['reviews'] + '\t\t\tComment: {}\n'.format(result['reviews'][i]['text'].encode('utf-8'))
				output['reviews'] = output['reviews'] + '\t\t\tSubmitted on: {}\n'.format(datetime.fromtimestamp(result['reviews'][i]['time']).strftime('%c'))
				# print '\t\t\tComment: {}'.format(result['reviews'][i]['text'].encode('utf-8'))
				# print '\t\t\tSubmitted on: {}\n'.format(datetime.fromtimestamp(result['reviews'][i]['time']).strftime('%c'))

		else:
			output['reviews'] = output['reviews'] + '\tNo reviews available for this place!\n'
			# print '\tNo reviews available for this place!'
		# if 'types' in result:
		# 	print 'Types: {}'.format(result['types'])
		if 'url' in result:
			output['url'] = result['url']
			# print 'Google URL: {}'.format(result['url'])
		if 'utc_offset' in result:
			output['utc_offset'] = float(result['utc_offset']) / 60
			# print 'UTC offset: {} hr'.format(float(result['utc_offset']) / 60)
		# if 'vicinity' in result:
		# 	print 'Vicinity: {}'.format(result['vicinity'])
		if 'website' in result:
			output['website'] = result['website'].encode('utf-8')
			# print 'Website: {}'.format(result['website'].encode('utf-8'))
	else:
		if myresponse['status'] == 'ZERO_RESULTS' or myresponse['status'] == 'NOT_FOUND':
			print 'No Results found!'
		if myresponse['status'] == 'UNKNOWN_ERROR':
			print 'Unknown error! Please try again.'
		if myresponse['status'] == 'OVER_QUERY_LIMIT':
			print 'Your over your quota of usage limit!'
		if myresponse['status'] == 'REQUEST_DENIED ':
			print 'Your request has been denied! Please verify your API key.'
		if myresponse['status'] == 'INVALID_REQUEST ':
			print 'Some required parameter is PROBABLY missing!'

	return output

def GetResponse(url, raw = False):
	"""
		Opens a passed URL and collects a JSON response.

		url -- URL whose response is to be read
		raw -- If True, then will return raw response from the URL, otherwise will deserealize the response to a Python object
	"""
	response = urllib.urlopen(url)
	jsonRaw = response.read()
	if raw:
		return jsonRaw
	jsonData = json.loads(jsonRaw)
	return jsonData



if __name__ == '__main__':
	# pulling up results for a query
	# TODO: take query as input
	myurl = build_url_text_search(query = 'top restaurants in Mayur Vihar Noida India')
	myresponse = GetResponse(myurl)
	myresults = GetResults(myresponse)
	if len(myresults) > 0:
		for result in myresults:
			print '{0}: {1}'.format(result['name'], result['address'])
		# printing the details of the top result
		placename = myresults[0]['name']
		myurl = build_url_place_details(placeid = myresults[0]['place_id'])
		myresponse = GetResponse(myurl)
		myresults = GetPlaceDetails(myresponse, review_count = 5)
		print '\n\nDetails of {}:\n'.format(placename)
		if len(myresults) > 0:
			for ele in myresults:
				print '{0}: {1}'.format(ele, myresults[ele])
	else:
		print 'No results available'
