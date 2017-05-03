import scrapy
import re
import scipy
import geocoder # geocoding library for retrieving latlon of attractions

"""
Call from terminal: 
scrapy crawl tripadvisor -o output.csv -s CLOSESPIDER_ITEMCOUNT=10
"""

class TripAdvisorSpider(scrapy.Spider):
	name = "tripadvisor" # name of spider when calling

	def __init__(self, search, numdata, *args, **kwargs):
		"""Spider for crawling TripAdvisor attractions for a 
		specified location.
		'search' - desired location to search, as str(location)
		'numdata' - number of data to parse before stopping"""
		super(TripAdvisorSpider, self).__init__(*args, **kwargs)
		self.search = search
		self.numdata = numdata
		self.count = 0
		self.urlbase = 'https://www.tripadvisor.com/'

	def start_requests(self):
		"""Search 'search' location on TripAdvisor attractions page."""
		urls = [ # search string url with self.search as str(location)
		'Search?redirect=true&q={0}&ssrc=g&type=eat'.format(self.search)
			]
		for url in urls: 
			url = self.urlbase + url # complete url with search string
			yield scrapy.Request(url=url, callback=self.parse_search) # search location

	def parse_search(self, response):
		"""Select first result in TripAdvisor search results page,
		and assume this location is the correct location."""
		info = response.xpath('//div[@class="result GEOS"]//div[@class="result_wrap "]/@onclick').extract()[0]
		ext = re.search(r".*Tourism-(.*?)-.*",info).group(1) # TripAdvisor location ID
		url = self.urlbase + 'Attractions-' + ext # TripAdvisor 'attractions' url
		yield scrapy.Request(url, callback=self.parse_list) # search attractions 

	def parse_list(self, response):
		"""Retrieve urls of all attractions, and follow those links. Continue on 
		to next page if minimum number of items have not been collected."""
		for href in response.xpath('//div[@class="property_title"]/a/@href').extract():
			url = response.urljoin(href) # urljoin incase TripAdvisor uses referenced urls
			if self.count < self.numdata: # keep collecting until have 'numdata' items
				# Follow attraction link (necessary to retrieve rating details)
				# priority=1 makes sure this occurs before continuing to next page (below)
				yield scrapy.Request(url, callback=self.parse_review, priority=1)

		# Comment to restrict search to first page only
		# next_page = response.xpath('//div[@class="unified pagination "]/a/@href').extract()[-1]
		# if next_page: # if another page exists... 
		# 	url = response.urljoin(next_page)
		# 	yield scrapy.Request(url, self.parse_list) # continue to next page

	def parse_review(self, response):
		"""Parse current attraction and retrieve attraction name, average rating,
		full address, and latlon coordinates."""
		self.count += 1 # 1 review is now being parsed, so add to counter

		# Pull all review data for attraction
		rating_info = response.xpath('//ul[@class="barChart"]')
		# check if this is a list of activities (necessary for activity types in 'attractions')
		if len(rating_info) == 0: 
			# Take first attraction from list
			href = response.xpath('//div[@class="property_title"]/a/@href').extract()[0]
			url = response.urljoin(href)
			yield scrapy.Request(url, callback=self.parse_review) # get attraction ratings
		else: 
			# Pull name of attraction
			attraction = response.xpath('//h1[@id="HEADING"]/text()').extract()[1]
			attraction = re.sub('\n','',attraction)
			
			# Pull ranking of attraction
			# ranking = response.xpath('//div[@class="slim_ranking"]//span[@class]/text()').extract()[0]
			# ranking = re.sub('#','',ranking)

			# Calculate average rating
			nums = rating_info.xpath('//div[@class="valueCount fr part"]/text()').extract()
			ratings = scipy.array([float(n) for n in nums if n])
			values = scipy.arange(5,0,-1)
			rateavg = scipy.sum(ratings*values)/scipy.sum(ratings)
			rateavg = str( scipy.around( rateavg, decimals=2 ) )

			# Pull address
			street = response.xpath('//span[@property="streetAddress"]/text()').extract()[0]
			city = response.xpath('//span[@property="addressLocality"]/text()').extract()[0]
			state = response.xpath('//span[@property="addressRegion"]/text()').extract()[0]
			postal = response.xpath('//span[@property="postalCode"]/text()').extract()[0]
			country = response.xpath('//span[@property="addressCountry"]/@content').extract()[0]
			address = street + ', ' + city + ', ' + state # + ' ' + postal + ' ' + country
			
			def latlon(address):
				return geocoder.google(address).latlng # geocode address
			
			# Attempt to geocode
			latlon = latlon(address) # geocode address
			if len(latlon) == 0: # if failed...
				address + ' ' + postal # add zipcode
				latlon = latlon(address) # geocode address
				if len(latlon) == 0: # if failed...
					address + ' ' + country # add country
					latlon = latlon(address) # geocode address

			# Return results
			yield {
				'attraction': attraction,
				# 'ranking': ranking,
				'rating': rateavg,
				'address': address,
				'latlon': latlon,
			}