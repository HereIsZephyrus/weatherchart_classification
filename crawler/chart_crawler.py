"""
Chart Crawler Module
"""
from typing import List
import os
from datetime import datetime, timedelta
import requests
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from .driver import Driver

def download_gallary_task(kind: str, urls: List[str]) -> None:
    """
    Download the gallary.
    """
    file_location = f"gallary/{kind}"
    chart_crawler = ChartCrawler(file_location)
    os.makedirs(file_location, exist_ok=True)
    for url in urls:
        chart_crawler.download_chart(url)

class ChartCrawler:
    """
    Chart Crawler class, download the chart webp from the url
    """
    def __init__(self, file_location: str):
        self.driver = Driver()
        self.file_location = file_location
        end_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.date_range = [
            (end_day - timedelta(hours=i * 6)).strftime('%Y%m%d%H%M')
            for i in range(30)
        ]
        self.projection_list = ["opencharts_eastern_asia",
                                "opencharts_eruasia",
                                "opencharts_south_east_asia_and_indonesia",
                                "opencharts_southern_asia"]

    def download_chart(self, base_url: str) -> None:
        """
        Download the chart webp from the url
        """
        for date in self.date_range:
            for projection in self.projection_list:
                url = f"{base_url}?base_time={date}&valid_time={date}&projection={projection}"
                self.driver.connect(url)
                self.driver.wait_for_update(timedelay=10)
                image_url = self.driver(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "img.jss511.jss288.jss516"))
                ).get_attribute("src")
                image_data = requests.get(image_url, timeout=10)
                with open(f"{self.file_location}/{date}_{projection}.webp", "wb") as handler:
                    handler.write(image_data.content)

    def __del__(self):
        self.driver = None # will call driver.__del__() when the reference is removed
