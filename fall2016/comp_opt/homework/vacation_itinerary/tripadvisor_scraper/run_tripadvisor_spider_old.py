# from twisted.internet import reactor
# from scrapy.crawler import Crawler
# from scrapy.conf import settings
# import logging
# import scrapy
from twisted.internet import reactor

# from scrapy.crawler import CrawlerProcess
from scrapy.crawler import Crawler
from scrapy.utils.project import get_project_settings
from tripadvisor_scraper.spiders.tripadvisor_spider import TripAdvisorSpider

from scrapy import signals
import logging
from scrapy.settings import Settings
from tripadvisor_scraper.pipelines import TripadvisorScraperPipeline

# Need to figure out how to save to local variable
def start_crawler(spider,search):
	# Set up spider
	spider = TripAdvisorSpider(search=search)

	# Set up settings
	settings = Settings()
	# settings.overrides['FEED_FORMAT']='csv'
	# settings.overrides['FEED_URI']='tripadvisor_{0}.csv'.format(search)
	settings.set('CLOSESPIDER_ITEMCOUNT',False)
	settings.set('ROBOTSTXT_OBEY',False)
	settings.set('COOKIES_ENABLED',False)
	settings.set('ITEM_PIPELINES',{
			'tripadvisor_scraper.pipelines.TripadvisorScraperPipeline': 300
		})
	settings.set('DOWNLOAD_DELAY',3)
	settings.set('LOG_FILENAME','log.log')
	# settings.overrides['LOG_FILENAME'] = 'log.log'
	# settings.overrides['ROBOTSTXT_OBEY'] = False # Ignore robots.txt
	# settings.overrides['CLOSESPIDER_ITEMCOUNT']=1
	# settings.overrides['DOWNLOAD_DELAY'] = 3
	# settings.overrides['COOKIES_ENABLED'] = False
	# settings.overrides['ITEM_PIPELINES'] = {
	#    'tripadvisor_scraper.pipelines.TripadvisorScraperPipeline': 300,
	# }

	# Set up crawler
	crawler = Crawler(spider, settings)
	# crawler.configure()
	crawler.signals.connect(spider_closed, signal=signals.spider_closed)
	crawler.crawl(spider)
	# crawler.start()
	# process.crawl(spider,search=search)
	# process.start()
	# crawler.start()
	# log.start()
	# reactor.run()

def spider_closed(spider):
	logging.msg('Spider closed: %s' % spider, level=log.INFO)
	print results

# 	spider = TripAdvisorSpider(domain=url) #domain=domain
# 	# settings = get_project_settings()
# 	# settings.overrides['LOG_FILE']='Log.log'
# 	crawler = Crawler(settings)
# 	crawler.configure()
# 	crawler.crawl(spider)
# 	crawler.start()

spider = 'tripadvisor'
search = 'austin'

start_crawler(spider,search)
# reactor.run()