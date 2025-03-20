import scrapy
import os

class TextSpider(scrapy.Spider):
    
    def start_requests(self):
        with open("urls.txt", "r") as f:
            urls = [line.strip() for line in f.readlines() if line.strip()]
        
        for i, url in enumerate(urls, start=0):
            yield scrapy.Request(url=url, callback=self.parse, meta={"file_index": i, "url": url})
    
    def parse(self, response):
        file_index = response.meta["file_index"]
        url = response.meta["url"]
        
        filename = f"pages/page_{file_index}.html"
        os.makedirs("pages", exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(response.text)
        
        # Записываем индекс
        with open("index.txt", "a", encoding="utf-8") as index_file:
            index_file.write(f"{file_index} {url}\n")
        
        self.log(f"Saved {filename}")
