from scrapy.crawler import CrawlerProcess
from text_spider import TextSpider

process = CrawlerProcess()
process.crawl(TextSpider)
process.start()