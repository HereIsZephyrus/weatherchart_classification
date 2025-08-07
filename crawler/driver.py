"""
Driver class that manages Chrome WebDriver and coordinates gallery operations.
"""
from typing import Optional
import logging
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, WebDriverException

logger = logging.getLogger(__name__)

class Driver:
    """
    Driver class that manages WebDriver and coordinates gallery operations.
    """
    def __init__(self, wait_timeout: int = 15, user_agent: Optional[str] = None):
        self.wait_timeout = wait_timeout
        self.user_agent = user_agent
        self.driver = None
        self.wait = None
        self.base_url = "https://charts.ecmwf.int/"
        self.session_data = {}
        self.setup_driver()

    def setup_driver(self) -> None:
        """
        Set up the Chrome WebDriver with optimized options.

        Returns:
            True if successful, False otherwise
        """
        try:
            chrome_options = Options()

            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')

            # Custom user agent
            if self.user_agent:
                chrome_options.add_argument(f'--user-agent={self.user_agent}')

            # Initialize driver
            self.driver = webdriver.Chrome(options=chrome_options)
            self.wait = WebDriverWait(self.driver, self.wait_timeout)
            logger.info("WebDriver and components initialized successfully")

        except WebDriverException as e:
            logger.error("Failed to setup WebDriver: %s", e)
    
    def __call__(self, operation, wait_time : Optional[int] = None) -> None:
        """
        Synchronize operations on the WebDriver.
        """
        try:
            self.wait.until(operation)
            if wait_time:
                time.sleep(wait_time)
        except TimeoutException:
            logger.error("Operation timed out after %d seconds", self.wait_timeout)
        except WebDriverException as e:
            logger.error("Error synchronizing operation: %s", e)

    def __del__(self):
        self.driver.quit()

    def get(self, url : str) -> None:
        """
        Get driver to a URL.
        """
        self.driver.get(url)

    def wait_for_update(self, timeout: int) -> None:
        """
        Wait for the gallery to update after filter changes.

        Args:
            timeout: Maximum time to wait for update
        """
        logger.debug("Waited for gallery update")
        time.sleep(timeout)  # Simple wait - could be improved with specific element checks
