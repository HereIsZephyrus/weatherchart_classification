"""
Gallery Crawler Module

This module provides functionality to crawl and parse the ECMWF Charts website
(https://charts.ecmwf.int/) to extract gallery information and chart data.
"""

import time
import logging
import json
from typing import List, Dict, Optional, Any
from urllib.parse import urljoin
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
from .driver import Driver

logger = logging.getLogger(__name__)

class GallaryCrawler:
    """
    A class to crawl and parse the ECMWF Charts gallery website.

    This class provides methods to:
    - Navigate to the ECMWF Charts website
    - Extract available chart categories and filters
    - Parse chart metadata and URLs
    """

    def __init__(self, driver: Driver):
        """
        Initialize the Gallery Crawler.

        Args:
            driver: existing WebDriver instance to use
            wait_timeout: Maximum time to wait for elements (seconds)
        """
        self.driver = driver
        self.main_layout = self.driver(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div.ReactVirtualized__Grid__innerScrollContainer")
            )
        )
        self.card_height = 520 # px
        self.gallary_line_count = None

    def navigate_to_gallery(self, base_url : str) -> bool:
        """
        Navigate to the ECMWF Charts gallery page.

        Args:
            url: The URL of the ECMWF Charts gallery page

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Navigating to ECMWF Charts: %s", base_url)
            self.driver.get(base_url)

            self.driver(EC.presence_of_element_located((By.TAG_NAME, "body")),3)

            logger.info("Successfully loaded ECMWF Charts page")
            return True

        except (TimeoutException, WebDriverException) as e:
            logger.error("Failed to navigate to gallery: %s", e)
            raise e

    def get_number_of_rows(self) -> int:
        """
        Get the number of rows in the gallery.
        """
        try:
            gallery_height = self.main_layout.get_attribute("style").split("height: ")[1].split("px")[0]
            logger.debug("Gallery height: %s", gallery_height)
            self.gallary_line_count = int(gallery_height) // self.card_height
            return self.gallary_line_count
        except (TimeoutException, WebDriverException) as e:
            logger.error("Error getting number of rows: %s", e)
            raise e

    def get_href_line(self, index: int) -> List[str]:
        """
        Get the href line for the current row count.
        """
        hrefs = []
        try:
            target_top = index * self.card_height
            logger.debug("Looking for row at top: %d (index: %d)", target_top, index)

            rows = self.main_layout.find_elements(By.CSS_SELECTOR, "div[style*='top:']")

            for row in rows:
                style = row.get_attribute("style")
                if f"top: {target_top}px" in style:
                    logger.debug("Found target row with top: %dpx", target_top)
                    links = row.find_elements(By.CSS_SELECTOR, "a[role='button']")
                    for link in links:
                        href = link.get_attribute("href")
                        if href:
                            hrefs.append(href)
                            logger.debug("Found href: %s", href)
                    break

            logger.debug("Total hrefs found: %d", len(hrefs))
            return hrefs

        except (NoSuchElementException, WebDriverException) as e:
            logger.error("Error getting href line for index %d: %s", index, e)
            return hrefs

    def scroll_to_row(self, row_count: int) -> None:
        """
        Scroll the gallery to the given row count.
        """
        try:
            scroll_top = row_count * self.card_height
            logger.debug("Scrolling to row %d, scroll_top: %d", row_count, scroll_top)

            self.driver.driver.execute_script(
                "arguments[0].scrollTop = arguments[1];", 
                self.main_layout,
                scroll_top
            )
            time.sleep(0.5)

        except WebDriverException as e:
            logger.error("Error scrolling to row %d: %s", row_count, e)
            raise e

    def get_next_row_count(self, index: int, current_row_count: int) -> int:
        """
        Get the next row count.
        """
        if (index < current_row_count + 3 - 1):
            return current_row_count

        if self.gallary_line_count is None:
            raise ValueError("Gallary line count not set")

        next_row_count = min(current_row_count + 3, self.gallary_line_count - 3)
        self.scroll_to_row(next_row_count)
        return next_row_count
