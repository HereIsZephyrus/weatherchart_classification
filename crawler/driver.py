"""
Driver class that manages Chrome WebDriver and coordinates gallery operations.
"""
from typing import Optional
import logging
import time
import os
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, WebDriverException

logger = logging.getLogger(__name__)

class Driver:
    """
    Driver class that manages WebDriver and coordinates gallery operations.
    """
    def __init__(self, wait_timeout: int = 30, user_agent: Optional[str] = None):
        self.wait_timeout = wait_timeout
        self.user_agent = user_agent
        self.driver = None
        self.wait = None
        self.base_url = "https://charts.ecmwf.int/"
        self.session_data = {}
        self.setup_driver()
        self.connect()

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
            chrome_options.add_argument('--window-size=1920,1080')

            # Custom user agent
            if self.user_agent:
                chrome_options.add_argument(f'--user-agent={self.user_agent}')

            # Initialize driver
            self.driver = webdriver.Chrome(options=chrome_options)
            self.wait = WebDriverWait(self.driver, self.wait_timeout)
            logger.info("WebDriver and components initialized successfully")

        except WebDriverException as e:
            logger.error("Failed to setup WebDriver: %s", e)
            raise e

    def __call__(self, operation, wait_time: Optional[int] = None):
        """
        Synchronize operations on the WebDriver.

        Returns:
            The result of the operation (usually a WebElement or list of WebElements)
        """
        try:
            result = self.wait.until(operation)
            if wait_time:
                time.sleep(wait_time)
            return result
        except TimeoutException as e:
            logger.error("Operation timed out after %d seconds", self.wait_timeout)
            raise e
        except WebDriverException as e:
            logger.error("Error synchronizing operation: %s", e)
            raise e

    def save_html(self, filename: Optional[str] = None) -> str:
        """
        Save current page HTML to file for debugging.

        Args:
            filename: Optional filename, if not provided, will use timestamp

        Returns:
            The filename of the saved HTML file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"debug_page_{timestamp}.html"

        try:
            # Create debug directory if it doesn't exist
            debug_dir = "debug_html"
            os.makedirs(debug_dir, exist_ok=True)

            filepath = os.path.join(debug_dir, filename)

            # Get page source and save to file
            page_source = self.driver.page_source
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(page_source)

            logger.info("HTML saved to: %s", filepath)
            return filepath

        except Exception as e:
            logger.error("Failed to save HTML: %s", e)
            raise e

    def __del__(self):
        if self.driver:
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

    def connect(self) -> bool:
        """
        Connect to the ECMWF Charts website.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.get(self.base_url)
            self.wait_for_update(10)
            return True

        except (TimeoutException, WebDriverException) as e:
            logger.error("Error connecting to remote gallery: %s", e)
            raise e
