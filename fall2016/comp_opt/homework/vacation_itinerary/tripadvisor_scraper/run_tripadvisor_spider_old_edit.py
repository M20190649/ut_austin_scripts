# from twisted.internet import reactor
# from scrapy.crawler import Crawler
# from scrapy.conf import settings
# import logging
# import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from tripadvisor_scraper.spiders.tripadvisor_spider import TripAdvisorSpider

class TACrawler:

	def __init__(self,search,itemcount=False,numdata=1000,dldelay=0,form='csv'):
		"""Calls tripadvisor_scraper spider to crawl tripadvisor 
		attractions page with provided search query and input settings
		'search' - search string for tripadvisor, must be str(location)
		'itemcount' - number of items to retrieve prior to stopping
		'numdata' - number of data items to retrieve prior to stopping
		'dldelay' - download delay (seconds), increase to about 2 or 3 to 
					minimize chances of being blocked
		'form' - format of resultant file output, can be 'csv' 'json',
				 'jsonlines', 'xml', 'pickle', or 'marshall'"""
		self.search = search
		if itemcount: self.itemcount = itemcount
		self.numdata = numdata
		self.form = form
		self.dldelay=dldelay
		spider = TripAdvisorSpider(search=search,numdata=numdata)
		self.spider = spider
		self.start_crawler()

# Need to figure out how to save to local variable
	def start_crawler(self):
		# Set up settings
		settings = get_project_settings()
		if self.form == 'var': 
			settings.set('ITEM_PIPELINES',{
					'tripadvisor_scraper.pipelines.TripadvisorScraperPipeline': 300
				})			
		if self.form == 'csv':
			settings.set('FEED_FORMAT','csv')
			settings.set('FEED_URI','{0}.csv'.format(self.search))
		if self.form == 'pickle':
			settings.set('FEED_FORMAT','pickle')
			settings.set('FEED_URI','{0}.pkl'.format(self.search))
		settings.set('ROBOTSTXT_OBEY',False)
		if self.itemcount: 
			settings.set('CLOSESPIDER_ITEMCOUNT',self.itemcount)
		settings.set('DOWNLOAD_DELAY',self.dldelay)
		settings.set('COOKIES_ENABLED',False)
		# settings.set('LOG_FILENAME','log.log')

		# Start crawler
		process = CrawlerProcess(settings)
		process.crawl(self.spider,search=self.search,numdata=self.numdata)
		process.start()

# 	spider = TripAdvisorSpider(domain=url) #domain=domain
# 	# settings = get_project_settings()
# 	# settings.overrides['LOG_FILE']='Log.log'
# 	crawler = Crawler(settings)
# 	crawler.configure()
# 	crawler.crawl(spider)
# 	crawler.start()

# spider = 'tripadvisor'
# search = 'san diego'

if __name__ == '__main__':

	TACrawler('san diego')

# start_crawler(spider,search)
# log.start()
# result = reactor.run()
# print result