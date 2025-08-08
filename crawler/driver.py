"""
Driver class that manages Chrome WebDriver and coordinates gallary operations.
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
    Driver class that manages WebDriver and coordinates gallary operations.
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
        except TimeoutException:
            logger.warning("Operation timed out after %d seconds", self.wait_timeout)
            return None
        except WebDriverException as e:
            logger.error("Error synchronizing operation: %s", e)
            raise e

    def save_html(self, filepath: Optional[str] = None) -> str:
        """
        Save current page HTML to file for debugging.

        Args:
            filename: Optional filename, if not provided, will use timestamp

        Returns:
            The filename of the saved HTML file
        """
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_dir = "debug_html"
            os.makedirs(debug_dir, exist_ok=True)
            filepath = f"debug_html/debug_page_{timestamp}.html"

        try:
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
        self.wait_for_page_load()

    def wait_for_update(self, timedelay: Optional[int] = None) -> None:
        """
        Wait for the gallary to update after filter changes.

        Args:
            timeout: Maximum time to wait for update
        """
        logger.debug("Waiting for page update and network idle")
        if timedelay is None:
            self.wait_for_page_load()
        else:
            time.sleep(timedelay)

    def connect(self, url: Optional[str] = None) -> bool:
        """
        Connect to the ECMWF Charts website.

        Returns:
            True if successful, False otherwise
        """
        try:
            if url:
                self.get(url)
            else:
                self.get(self.base_url)
            return True

        except (TimeoutException, WebDriverException) as e:
            logger.error("Error connecting to remote gallary: %s", e)
            raise e

    def wait_for_page_load(self, timeout: Optional[int] = None) -> None:
        """
        Wait until the page reports it has fully loaded.

        This waits for document.readyState === 'complete' and, if jQuery is present,
        waits until there are no active AJAX requests.

        Args:
            timeout: Optional override for max wait time in seconds
        """
        max_wait = timeout if timeout is not None else self.wait_timeout

        # Wait for DOM ready
        WebDriverWait(self.driver, max_wait).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )

        # If jQuery is present, wait for AJAX to settle (best-effort)
        try:
            WebDriverWait(self.driver, min(5, max_wait)).until(
                lambda d: d.execute_script(
                    "return (window.jQuery && jQuery.active === 0) || !window.jQuery;"
                )
            )
        except TimeoutException:
            logger.debug("jQuery idle wait timed out or jQuery not present")
