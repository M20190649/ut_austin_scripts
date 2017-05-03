from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from tripadvisor_scraper.spiders.tripadvisor_spider import TripAdvisorSpider

class TACrawler:

	def __init__(self,search,numdata=1000,dldelay=0,form='csv'):
		"""Calls tripadvisor_scraper spider to crawl tripadvisor 
		attractions page with provided search query and input settings.
		'search' - search string for tripadvisor, must be str(location)
		'itemcount' - number of items to retrieve prior to stopping
		'numdata' - number of data items to retrieve prior to stopping
		'dldelay' - download delay (seconds), increase to about 2 or 3 to 
					minimize chances of being blocked
		'form' - format of resultant file output; 'csv','var', or 'pickle
		'"""
		self.search = search
		self.numdata = numdata
		self.form = form
		self.dldelay = dldelay
		spider = TripAdvisorSpider(search=search,numdata=numdata) # assign spider
		self.spider = spider
		self.start_crawler() # set spider settings and start crawling

	def start_crawler(self):
		"""Start crawling specified spider."""
		# Set up settings
		settings = get_project_settings() # retrieve settings specified in settings.py
		if self.form == 'var': # store output as local variable; does not work
			settings.set('ITEM_PIPELINES',{
					'tripadvisor_scraper.pipelines.TripadvisorScraperPipeline': 300
				})			
		if self.form == 'csv': # settings for csv output option
			settings.set('FEED_FORMAT','csv')
			settings.set('FEED_URI','{0}.csv'.format(self.search))
		if self.form == 'pickle': # settings for pickle output option
			settings.set('FEED_FORMAT','pickle')
			settings.set('FEED_URI','{0}.pkl'.format(self.search))
		settings.set('ROBOTSTXT_OBEY',False) # ignore robots.txt
		settings.set('CLOSESPIDER_ITEMCOUNT',self.numdata) # minimize max downloads
		settings.set('DOWNLOAD_DELAY',self.dldelay) # delay between downloads
		settings.set('COOKIES_ENABLED',False) # do not save cookies
		# import logging; settings.set('LOG_FILENAME','log.log') # enable logging

		# Start crawler
		process = CrawlerProcess(settings)
		process.crawl(self.spider,search=self.search,numdata=self.numdata)
		process.start() # start crawling