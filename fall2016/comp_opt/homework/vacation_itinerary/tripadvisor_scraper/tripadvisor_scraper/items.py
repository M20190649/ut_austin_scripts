import scrapy

class TripAdvisorReviewItem(scrapy.Item):
	attraction = scrapy.Field()
	rating = scrapy.Field()
	ranking = scrapy.Field()
	address = scrapy.Field()
	latlon = scrapy.Field()