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
        self.chart_metadata = {}

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
            return False

    def extract_chart_metadata(self, max_charts: int = 50) -> List[Dict[str, Any]]:
        """
        Extract metadata for available charts.

        Args:
            max_charts: Maximum number of charts to extract metadata for

        Returns:
            List of chart metadata dictionaries
        """
        chart_data = []

        try:
            # Look for chart elements or gallery items
            chart_selectors = [
                ".chart-item",
                ".gallery-item",
                ".chart-card",
                "[data-chart-id]",
                ".product-item",
                "img[src*='chart']",
                "img[src*='forecast']"
            ]

            chart_elements = []
            for selector in chart_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        chart_elements.extend(elements[:max_charts])
                        break
                except NoSuchElementException:
                    continue

            if not chart_elements:
                logger.warning("No chart elements found on the page")
                return chart_data

            logger.info("Found %d chart elements", len(chart_elements))

            for i, element in enumerate(chart_elements[:max_charts]):
                try:
                    metadata = self._extract_single_chart_metadata(element, i)
                    if metadata:
                        chart_data.append(metadata)

                except (TimeoutException, WebDriverException) as e:
                    logger.debug("Error extracting metadata for chart %d: %s", i, e)
                    continue

            self.chart_metadata = chart_data
            logger.info("Successfully extracted metadata for %d charts", len(chart_data))
            return chart_data

        except (TimeoutException, WebDriverException) as e:
            logger.error("Error extracting chart metadata: %s", e)
            return chart_data

    def _extract_single_chart_metadata(self, element, index: int) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from a single chart element.

        Args:
            element: WebElement representing a chart
            index: Chart index for identification

        Returns:
            Dictionary with chart metadata or None if extraction fails
        """
        try:
            metadata = {
                'index': index,
                'title': '',
                'description': '',
                'image_url': '',
                'data_url': '',
                'parameters': [],
                'forecast_type': '',
                'valid_time': '',
                'lead_time': '',
                'region': '',
                'element_html': ''
            }

            # Extract title
            title_selectors = [
                '.chart-title', '.title', 'h1', 'h2', 'h3', 
                '[title]', 'alt', '.product-name'
            ]

            for selector in title_selectors:
                try:
                    title_element = element.find_element(By.CSS_SELECTOR, selector)
                    metadata['title'] = title_element.text.strip() or title_element.get_attribute('title') or title_element.get_attribute('alt')
                    if metadata['title']:
                        break
                except NoSuchElementException:
                    continue

            # Extract image URL
            try:
                img_element = element.find_element(By.TAG_NAME, 'img')
                src = img_element.get_attribute('src')
                if src:
                    metadata['image_url'] = urljoin(self.driver.base_url, src)
            except NoSuchElementException:
                pass

            # Extract description
            desc_selectors = ['.description', '.chart-desc', 'p', '.subtitle']
            for selector in desc_selectors:
                try:
                    desc_element = element.find_element(By.CSS_SELECTOR, selector)
                    metadata['description'] = desc_element.text.strip()
                    if metadata['description']:
                        break
                except NoSuchElementException:
                    continue

            # Extract data URL (if available)
            try:
                link_element = element.find_element(By.TAG_NAME, 'a')
                href = link_element.get_attribute('href')
                if href:
                    metadata['data_url'] = urljoin(self.driver.base_url, href)
            except NoSuchElementException:
                pass

            # Store raw HTML for further analysis
            metadata['element_html'] = element.get_attribute('outerHTML')[:500]  # Truncate for storage

            # Parse parameters and forecast type from title/description
            self._parse_chart_attributes(metadata)

            return metadata if metadata['title'] or metadata['image_url'] else None

        except (TimeoutException, WebDriverException) as e:
            logger.debug("Error extracting metadata for chart element: %s", e)
            return None

    def _parse_chart_attributes(self, metadata: Dict[str, Any]) -> None:
        """
        Parse chart attributes from title and description.

        Args:
            metadata: Chart metadata dictionary to update
        """
        text = (metadata.get('title', '') + ' ' + metadata.get('description', '')).lower()

        # Extract parameters
        parameter_keywords = [
            'wind', 'temperature', 'precipitation', 'pressure', 'cloud',
            'humidity', 'geopotential', 'snow', 'ocean', 'water vapour'
        ]

        for param in parameter_keywords:
            if param in text:
                metadata['parameters'].append(param.title())

        # Extract forecast type
        if 'ensemble' in text:
            metadata['forecast_type'] = 'Ensemble'
        elif 'control' in text or 'hres' in text:
            metadata['forecast_type'] = 'Control'
        elif 'aifs' in text:
            metadata['forecast_type'] = 'AIFS'

        # Extract region
        region_keywords = ['europe', 'global', 'atlantic', 'pacific', 'arctic']
        for region in region_keywords:
            if region in text:
                metadata['region'] = region.title()
                break

    def get_chart_image_urls(self) -> List[str]:
        """
        Get URLs of all chart images on the current page.

        Args:
            base_url: The base URL of the ECMWF Charts website

        Returns:
            List of image URLs
        """
        image_urls = []

        try:
            img_elements = self.driver.find_elements(By.TAG_NAME, 'img')

            for img in img_elements:
                src = img.get_attribute('src')
                if src and any(keyword in src.lower() for keyword in ['chart', 'forecast', 'weather', 'plot']):
                    full_url = urljoin(self.driver.base_url, src)
                    image_urls.append(full_url)

            logger.info("Found %d chart image URLs", len(image_urls))
            return list(set(image_urls))  # Remove duplicates

        except (TimeoutException, WebDriverException) as e:
            logger.error("Error extracting image URLs: %s", e)
            return image_urls

    def navigate_to_specific_chart_type(self, chart_type: str) -> bool:
        """
        Navigate to a specific type of charts (e.g., medium-range, seasonal).

        Args:
            chart_type: Type of chart to navigate to

        Returns:
            True if navigation successful, False otherwise
        """
        try:
            # Look for navigation links or buttons
            nav_selectors = [
                f"a[href*='{chart_type}']",
                f"//a[contains(text(), '{chart_type}')]",
                f".nav-link:contains('{chart_type}')",
                f"button:contains('{chart_type}')"
            ]

            for selector in nav_selectors:
                try:
                    if selector.startswith('//'):
                        element = self.driver.find_element(By.XPATH, selector)
                    else:
                        element = self.driver.find_element(By.CSS_SELECTOR, selector)

                    element.click()
                    time.sleep(2)
                    logger.info("Navigated to %s charts", chart_type)
                    return True

                except NoSuchElementException:
                    continue

            logger.warning("Could not find navigation for chart type: %s", chart_type)
            return False

        except (TimeoutException, WebDriverException) as e:
            logger.error("Error navigating to chart type %s: %s", chart_type, e)
            return False

    def get_page_info(self) -> Dict[str, Any]:
        """
        Get general information about the current page.

        Returns:
            Dictionary with page information
        """
        info = {
            'url': self.driver.current_url,
            'title': '',
            'description': '',
            'available_sections': [],
            'total_charts': 0,
        }

        try:
            # Get page title
            info['title'] = self.driver.title

            # Check for meta description
            try:
                meta_desc = self.driver.find_element(By.CSS_SELECTOR, 'meta[name="description"]')
                info['description'] = meta_desc.get_attribute('content')
            except NoSuchElementException:
                pass

            # Find main sections
            section_selectors = ['nav a', '.section-title', 'h1', 'h2', '.menu-item']
            for selector in section_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        text = element.text.strip()
                        if text and len(text) < 100:  # Reasonable section title length
                            info['available_sections'].append(text)
                except NoSuchElementException:
                    continue

            # Count charts
            chart_elements = self.driver.find_elements(By.CSS_SELECTOR, 'img, .chart-item, .gallery-item')
            info['total_charts'] = len(chart_elements)

            return info

        except (TimeoutException, WebDriverException) as e:
            logger.error("Error getting page info: %s", e)
            return info

    def save_metadata_to_file(self, filename: str) -> bool:
        """
        Save extracted metadata to a JSON file.

        Args:
            filename: Output filename

        Returns:
            True if successful, False otherwise
        """
        try:
            data = {
                'page_info': self.get_page_info(),
                'chart_metadata': self.chart_metadata,
                'extraction_timestamp': time.time()
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info("Metadata saved to %s", filename)
            return True

        except (TimeoutException, WebDriverException) as e:
            logger.error("Error saving metadata to file: %s", e)
            return False
