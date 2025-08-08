"""
Main Crawler Module

This module provides the main Crawler class that coordinates WebDriver management
and integrates GallerySelector and GallaryCrawler functionality for comprehensive
weather chart gallery crawling and filtering operations.

Author: AI Assistant
"""

import time
import logging
from typing import Dict, List, Any, Optional
from selenium.common.exceptions import TimeoutException, WebDriverException
from .driver import Driver
from .gallary_selector import GallerySelector
from .gallary_crawler import GallaryCrawler

logger = logging.getLogger(__name__)

class Crawler:
    """
    Main Crawler class that manages WebDriver and coordinates gallery operations.

    This class provides:
    - Centralized WebDriver management
    - Integration of GallerySelector for local HTML filtering
    - Integration of GallaryCrawler for live website crawling
    - Unified interface for both local and remote operations
    - Session management and resource cleanup
    """

    def __init__(self):
        """
        Initialize the main Crawler.

        Args:
            headless: Whether to run browser in headless mode
            wait_timeout: Maximum time to wait for elements (seconds)
            user_agent: Custom user agent string
        """
        self.driver = Driver()
        self.gallery_selector = GallerySelector(self.driver)
        self.gallery_crawler = GallaryCrawler(self.driver)

    def apply_filters(self, filters: Dict[str, List[str]]) -> None:
        """
        Apply filters to the current gallery (local or remote).

        Args:
            filters: Dictionary with filter categories and values

        Returns:
            Dictionary with filter application results
        """
        if not self.gallery_selector:
            logger.error("Gallery selector not initialized")

        try:
            self.gallery_selector.apply_filters(filters)
        except (TimeoutException, WebDriverException) as e:
            logger.error("Error applying filters: %s", e)
            raise e

    def extract_chart_hrefs(self) -> List[str]:
        """
        Extract product hrefs from the current gallery.

        Args:
            max_items: Maximum number of items to extract

        Returns:
            List of product hrefs
        """

        href_list = []
        try:
            row_num = self.gallery_crawler.get_number_of_rows()
            if row_num == 0:
                logger.warning("No rows found in the gallery")
                return href_list
        except (TimeoutException, WebDriverException) as e:
            logger.error("Error getting number of rows: %s", e)
            raise e

        current_row_count = 0
        try:
            for index in range(row_num):
                href_line = self.gallery_crawler.get_href_line(index)
                href_list.extend(href_line)
                current_row_count = self.gallery_crawler.get_next_row_count(index, current_row_count)
        except (TimeoutException, WebDriverException) as e:
            logger.error("Error getting chart hrefs: %s", e)
            raise e
        return href_list

    def __del__(self):
        self.driver = None
        self.gallery_selector = None
        self.gallery_crawler = None

    def filter(self, params: List[str]) -> None:
        """
        Filter the gallery by a parameter.
        """
        self.gallery_selector.filter(params)

    def download(self, filename: Optional[str] = None) -> None:
        """
        Download the gallery.
        """
        self.driver.save_html(filename)
