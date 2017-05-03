# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

# class TripadvisorScraperPipeline(object):
#     def process_item(self, item, spider):
#         return item

import results

class TripadvisorScraperPipeline(object):
	def process_item(self, item, spider):
		ta_data.results.append(dict(item))

	# results = []
	# def close_spider(spider):
	# 	print results
