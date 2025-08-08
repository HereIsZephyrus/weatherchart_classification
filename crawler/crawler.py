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

    def extract_gallery_metadata(self, max_items: int = 50) -> Dict[str, Any]:
        """
        Extract metadata from the current gallery.

        Args:
            max_items: Maximum number of items to extract

        Returns:
            Dictionary with extracted metadata
        """
        metadata = {
            'timestamp': time.time(),
            'filters': {},
            'charts': [],
            'page_info': {}
        }

        try:
            if self.gallery_selector:
                metadata['current_filter_states'] = self.gallery_selector.get_current_filter_states()

            elif self.gallery_crawler:
                if self.gallery_crawler:
                    metadata['page_info'] = self.gallery_crawler.get_page_info()
                    metadata['charts'] = self.gallery_crawler.extract_chart_metadata(max_items)
                    metadata['image_urls'] = self.gallery_crawler.get_chart_image_urls()

            logger.info("Extracted metadata for %s gallery", self.driver.base_url)
            return metadata

        except (TimeoutException, WebDriverException) as e:
            logger.error("Error extracting metadata: %s", e)
            raise e

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