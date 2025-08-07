"""
Gallery Selector Module

This module provides functionality to interact with gallery filter checkboxes
for weather chart classification. It can click specific checkboxes to filter
the gallery based on different criteria like surface/atmosphere, product types,
and parameters.

Author: AI Assistant
"""

import time
import logging
from typing import List, Dict, Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GallerySelector:
    """
    A class to handle gallery filtering through checkbox interactions.

    This class provides methods to:
    - Initialize a web driver
    - Click specific checkboxes by value or text
    - Get current filter states
    - Apply multiple filters at once
    """

    def __init__(self, driver: webdriver.Chrome, headless: bool = True, wait_timeout: int = 10):
        """
        Initialize the Gallery Selector.

        Args:
            headless: Whether to run browser in headless mode
            wait_timeout: Maximum time to wait for elements (seconds)
        """
        self.wait_timeout = wait_timeout
        self.driver = driver
        self.wait = WebDriverWait(self.driver, self.wait_timeout)
        self.headless = headless

        # Filter categories mapping
        self.filter_categories = {
            'surface_atmosphere': {
                'Surface': 'Surface',
                'Atmosphere': 'Atmosphere'
            },
            'product_type': {
                'Control Forecast (ex-HRES)': 'Control Forecast (ex-HRES)',
                'Ensemble forecast (ENS)': 'Ensemble forecast (ENS)',
                'Extreme forecast index': 'Extreme forecast index',
                'Point-based products': 'Point-based products',
                'AIFS Single': 'AIFS Single',
                'AIFS Ensemble forecast': 'AIFS Ensemble forecast',
                'AIFS ENS Control': 'AIFS ENS Control',
                'Experimental: Machine learning models': 'Experimental: Machine learning models',
                'Atmospheric composition': 'Atmospheric composition'
            },
            'parameters': {
                'Wind': 'Wind',
                'Mean sea level pressure': 'Mean sea level pressure',
                'Temperature': 'Temperature',
                'Geopotential': 'Geopotential',
                'Precipitation': 'Precipitation',
                'Cloud': 'Cloud',
                'Water vapour': 'Water vapour',
                'Humidity': 'Humidity',
                'Indices': 'Indices',
                'Snow': 'Snow',
                'Ocean waves': 'Ocean waves',
                'Surface characteristics': 'Surface characteristics'
            }
        }

    def find_checkbox_by_value(self, value: str) -> Optional[object]:
        """
        Find a checkbox input element by its value attribute.

        Args:
            value: The value attribute of the checkbox

        Returns:
            WebElement if found, None otherwise
        """
        try:
            checkbox = self.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, f'input.jss5663[value="{value}"]'))
            )
            return checkbox
        except TimeoutException:
            logger.warning("Checkbox with value '%s' not found or not clickable", value)
            return None

    def click_checkbox(self, value: str) -> bool:
        """
        Click a checkbox by its value.

        Args:
            value: The value attribute of the checkbox to click

        Returns:
            True if successfully clicked, False otherwise
        """
        try:
            checkbox = self.find_checkbox_by_value(value)
            if checkbox:
                # Scroll to element to ensure it's visible
                self.driver.execute_script("arguments[0].scrollIntoView(true);", checkbox)
                time.sleep(0.5)

                checkbox.click()
                logger.info("Successfully clicked checkbox: %s", value)
                return True

            logger.error("Could not find checkbox with value: %s", value)
            return False
        except (TimeoutException, AttributeError, WebDriverException) as e:
            logger.error("Error clicking checkbox '%s': %s", value, e)
            return False

    def get_checkbox_state(self, value: str) -> Optional[bool]:
        """
        Get the current state (checked/unchecked) of a checkbox.

        Args:
            value: The value attribute of the checkbox

        Returns:
            True if checked, False if unchecked, None if not found
        """
        try:
            checkbox = self.find_checkbox_by_value(value)
            if checkbox:
                return checkbox.is_selected()
            return None
        except (TimeoutException, AttributeError, WebDriverException) as e:
            logger.error("Error getting checkbox state for '%s': %s", value, e)
            return None

    def apply_filters(self, filters: Dict[str, List[str]]) -> Dict[str, bool]:
        """
        Apply multiple filters by clicking checkboxes.

        Args:
            filters: Dictionary with category as key and list of values as filters
                    Example: {
                        'parameters': ['Wind', 'Temperature'],
                        'product_type': ['Control Forecast (ex-HRES)']
                    }

        Returns:
            Dictionary with filter values as keys and success status as values
        """
        results = {}

        for category, values in filters.items():
            if category not in self.filter_categories:
                logger.warning("Unknown filter category: %s", category)
                continue

            for value in values:
                if value in self.filter_categories[category]:
                    success = self.click_checkbox(value)
                    results[value] = success
                else:
                    logger.warning("Unknown filter value '%s' in category '%s'", value, category)
                    results[value] = False

        return results

    def get_all_available_filters(self) -> Dict[str, List[str]]:
        """
        Get all available filter options.

        Returns:
            Dictionary with categories and their available options
        """
        return self.filter_categories.copy()

    def get_current_filter_states(self) -> Dict[str, Dict[str, bool]]:
        """
        Get the current state of all filter checkboxes.

        Returns:
            Nested dictionary with categories and checkbox states
        """
        states = {}

        for category, options in self.filter_categories.items():
            states[category] = {}
            for option_name, option_value in options.items():
                state = self.get_checkbox_state(option_value)
                states[category][option_name] = state

        return states

    def clear_all_filters(self) -> Dict[str, bool]:
        """
        Uncheck all currently checked filters.

        Returns:
            Dictionary with results of unchecking operations
        """
        results = {}
        current_states = self.get_current_filter_states()

        for category, options in current_states.items():
            for option_name, is_checked in options.items():
                if is_checked:  # Only uncheck if currently checked
                    option_value = self.filter_categories[category][option_name]
                    success = self.click_checkbox(option_value)
                    results[option_value] = success

        return results

    def wait_for_gallery_update(self, timeout: int = 5) -> None:
        """
        Wait for the gallery to update after filter changes.

        Args:
            timeout: Maximum time to wait for update
        """
        time.sleep(timeout)  # Simple wait - could be improved with specific element checks
        logger.info("Waited for gallery update")

    def close(self) -> None:
        """Close the web driver and clean up resources."""
        if self.driver:
            self.driver.quit()
            logger.info("WebDriver closed")
        else:
            logger.info("WebDriver cleanup skipped (shared instance)")
