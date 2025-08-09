"""
Chart Crawler Module
"""
from typing import List
import os
from datetime import datetime, timedelta
import logging
import requests
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from .driver import Driver

logger = logging.getLogger(__name__)

def download_gallery_task(kind: str, urls: List[str]) -> None:
    """
    Download the gallery.
    """
    file_location = f"gallery/{kind}"
    chart_crawler = ChartCrawler(file_location)
    os.makedirs(file_location, exist_ok=True)
    for url in urls:
        chart_crawler.download_chart(url)
    logger.info("Downloaded charts for %s.", kind)

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
        dataset_name = base_url.split("/")[-1]
        for date in self.date_range:
            for projection in self.projection_list:
                file_name = f"{dataset_name}_{date}_{projection}.webp"
                if os.path.exists(f"{self.file_location}/{file_name}"):
                    logger.info("Chart %s already exists.", file_name)
                    continue

                url = f"{base_url}?base_time={date}&valid_time={date}&projection={projection}"
                self.driver.connect(url)
                self.driver.wait_for_update(timedelay=10)
                try:
                    image_element = self.driver(
                        EC.presence_of_element_located((By.TAG_NAME, "img"))
                    )
                except TimeoutException:
                    logger.warning("Failed to get image url for %s.", url)
                    continue

                if image_element is None:
                    logger.warning("No image found for %s.", url)
                    continue
                image_url = image_element.get_attribute("src")

                if image_url is None:
                    logger.warning("Failed to get image url for %s.", url)
                    continue

                try:
                    image_data = requests.get(image_url, timeout=10)
                except requests.exceptions.RequestException:
                    logger.warning("Failed to get image data for %s.", image_url)
                    continue

                with open(f"{self.file_location}/{file_name}", "wb") as handler:
                    handler.write(image_data.content)
                logger.info("Downloaded chart for %s.", url)

    def __del__(self):
        self.driver = None # will call driver.__del__() when the reference is removed
