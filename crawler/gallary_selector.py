"""
Gallery Selector Module

This module provides functionality to interact with gallary filter checkboxes
for weather chart classification. It can click specific checkboxes to filter
the gallary based on different criteria like surface/atmosphere, product types,
and parameters.
"""
import logging
from typing import List, Dict
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from .driver import Driver

logger = logging.getLogger(__name__)

class GallerySelector:
    """
    A class to handle gallary filtering through checkbox interactions.

    This class provides methods to:
    - Initialize a web driver
    - Click specific checkboxes by value or text
    - Get current filter states
    - Apply multiple filters at once
    """

    def __init__(self, driver : Driver):
        """
        Initialize the Gallery Selector.
        """
        self.driver = driver
        self.filter_categories = None
        self.filter_orientation = None
        self.init_filter()

    def init_filter(self) -> None:
        """
        Initialize the filter categories.
        """
        self.filter_categories = {}
        self.filter_orientation = {}

        try:
            filter_divs = self.driver(
                    EC.presence_of_all_elements_located(
                        (By.CSS_SELECTOR, "div.MuiFormGroup-root-276.jss210")
                    )
                )
        except TimeoutException as e:
            logger.warning("Filter divs not found")
            raise e
        except WebDriverException as e:
            logger.error("Error finding filter divs: %s", e)
            raise e

        try:
            for parent_div in filter_divs:
                legend_name = parent_div.find_element(
                    By.CSS_SELECTOR, "legend.MuiFormLabel-root-278.jss211").text
                self.filter_categories[legend_name] = []
                self.filter_orientation[legend_name] = {}
                category = self.filter_categories[legend_name]
                orientation = self.filter_orientation[legend_name]
                checkboxes = parent_div.find_elements(By.CSS_SELECTOR, "input.jss294")
                for checkbox in checkboxes:
                    category.append(checkbox.get_attribute("value"))
                    orientation[checkbox.get_attribute("value")] = checkbox

        except TimeoutException as e:
            logger.warning("Filter divs not found")
            raise e
        except WebDriverException as e:
            logger.error("Error finding filter divs: %s", e)
            raise e

        logger.debug(self.filter_categories)
        logger.debug(self.filter_orientation)
        logger.info("Filter categories initialized")

    def click_checkbox(self, checkbox: WebElement) -> None:
        """
        Click a checkbox by its value.
        """
        try:
            checkbox.click()
        except TimeoutException as e:
            logger.warning("Checkbox with value '%s' not found", checkbox.get_attribute("value"))
            raise e
        except WebDriverException as e:
            logger.error("Error clicking checkbox: %s", e)
            raise e

    def get_checkbox_state(self, checkbox: WebElement) -> bool:
        """
        Get the state of a checkbox.
        """
        try:
            return checkbox.is_selected()
        except TimeoutException as e:
            logger.warning("Checkbox with value '%s' not found", checkbox.get_attribute("value"))
            raise e
        except WebDriverException as e:
            logger.error("Error getting checkbox state: %s", e)
            raise e

    def filter_on(self, checkbox: WebElement) -> None:
        """
        Filter on a checkbox.
        """
        if not self.get_checkbox_state(checkbox):
            self.click_checkbox(checkbox)

    def filter_off(self, checkbox: WebElement) -> None:
        """
        Filter off a checkbox.
        """
        if self.get_checkbox_state(checkbox):
            self.click_checkbox(checkbox)

    def apply_filters(self, filters: Dict[str, List[str]]) -> None:
        """
        Apply multiple filters by clicking checkboxes.

        Args:
            filters: Dictionary with category as key and list of values as filters
                    Example: {
                        'Parameters': ['Wind', 'Temperature'],
                        'Product type': ['Control Forecast (ex-HRES)']
                    }
        """

        for category, values in self.filter_categories.items():
            for value in values:
                checkbox : WebElement = self.filter_orientation[category][value]
                if value in filters[category]:
                    self.filter_on(checkbox)
                else:
                    self.filter_off(checkbox)

    def get_current_filter_states(self) -> Dict[str, Dict[str, bool]]:
        """
        Get the current state of all filter checkboxes.

        Returns:
            Nested dictionary with categories and checkbox states
        """
        states = {}

        for category, options in self.filter_orientation.items():
            states[category] = {}
            for checkbox in options:
                value = checkbox.get_attribute("value")
                states[category][value] = self.get_checkbox_state(checkbox)

        return states

    def clear_all_filters(self) -> None:
        """
        Uncheck all currently checked filters.
        """
        for options in self.filter_orientation.values():
            for checkbox in options:
                self.filter_off(checkbox)

    def filter(self, params: List[str]) -> None:
        """
        Filter the gallary by a parameter.
        """
        self.apply_filters(self._build_filter_categories(params))

    def _build_filter_categories(self, params: List[str]) -> Dict[str, List[str]]:
        """
        Build the filter categories from the parameters.
        """
        filter_categories = {
            "Range": ["Medium (15 days)"],
            "Type": ["Forecasts"],
            "Component": ['Surface', 'Atmosphere'],
            "Product type":['Control Forecast (ex-HRES)',
                            'Ensemble forecast (ENS)',
                            'Extreme forecast index',
                            'AIFS Single',
                            'AIFS Ensemble forecast',
                            'AIFS ENS Control'],
        }
        filter_categories["Parameters"] = params
        return filter_categories
