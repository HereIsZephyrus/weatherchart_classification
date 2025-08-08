"""
Main Crawler Module

This module provides the main Crawler class that coordinates WebDriver management
and integrates GallerySelector and GallaryCrawler functionality for comprehensive
weather chart gallary crawling and filtering operations.

Author: AI Assistant
"""

import logging
from typing import Dict, List, Optional, Set
from selenium.common.exceptions import TimeoutException, WebDriverException
from .driver import Driver
from .gallary_selector import GallerySelector
from .gallary_crawler import GallaryCrawler

logger = logging.getLogger(__name__)

class Crawler:
    """
    Main Crawler class that manages WebDriver and coordinates gallary operations.

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
        Apply filters to the current gallary (local or remote).

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
        Extract product hrefs from the current gallary.

        Args:
            max_items: Maximum number of items to extract

        Returns:
            List of product hrefs
        """

        href_list = []
        try:
            row_num = self.gallery_crawler.get_number_of_rows()
            if row_num == 0:
                logger.warning("No rows found in the gallary")
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
        Filter the gallary by a parameter.
        """
        self.gallery_selector.filter(params)

    def download_page(self, filename: Optional[str] = None) -> None:
        """
        Download the gallary.
        """
        self.driver.save_html(filename)

    def reorganize_gallery(self, gallary: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Reorganize gallary data to group URLs that appear in multiple parameters.
        
        Args:
            gallary: Original gallary dictionary, keys are parameter names, values are URL lists
            
        Returns:
            Reorganized gallary dictionary, ensuring each URL belongs to only one type
        """
        # standardize the parameter names first
        standardized_gallery = {}
        for param in gallary.keys():
            new_param = param.lower()
            new_param = new_param.replace(" ", "_") 
            new_param = new_param.replace("-", "_")
            standardized_gallery[new_param] = gallary[param]
        gallary = standardized_gallery
        logger.info("standardized gallary")

        url_to_params: Dict[str, Set[str]] = {}

        # Count how many times each URL appears in which parameters
        for param, urls in gallary.items():
            for url in urls:
                if url not in url_to_params:
                    url_to_params[url] = set()
                url_to_params[url].add(param)

        new_gallery: Dict[str, List[str]] = {}

        for url, params in url_to_params.items():
            if len(params) == 1:
                # Only appears in one parameter, keep original classification
                param = list(params)[0]
                if param not in new_gallery:
                    new_gallery[param] = []
                new_gallery[param].append(url)
            else:
                # Appears in multiple parameters, create combined classification
                combined_key = 'A'.join(sorted(params))
                if combined_key not in new_gallery:
                    new_gallery[combined_key] = []
                new_gallery[combined_key].append(url)

        logger.info("reorganized gallary")
        return new_gallery
