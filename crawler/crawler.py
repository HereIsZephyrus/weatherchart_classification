"""
Main Crawler Module

This module provides the main Crawler class that coordinates WebDriver management
and integrates GallerySelector and GallaryCrawler functionality for comprehensive
weather chart gallery crawling and filtering operations.

Author: AI Assistant
"""

import time
import logging
import json
from typing import Dict, List, Optional, Any
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, WebDriverException

# Import our custom modules
try:
    from .selector import GallerySelector
    from .gallary_crawler import GallaryCrawler
except ImportError:
    from selector import GallerySelector
    from gallary_crawler import GallaryCrawler

# Configure logging
logging.basicConfig(level=logging.INFO)
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

    def __init__(self,
                 headless: bool = True,
                 wait_timeout: int = 15,
                 window_size: str = "1920,1080",
                 user_agent: Optional[str] = None):
        """
        Initialize the main Crawler.

        Args:
            headless: Whether to run browser in headless mode
            wait_timeout: Maximum time to wait for elements (seconds)
            window_size: Browser window size as "width,height"
            user_agent: Custom user agent string
        """
        self.headless = headless
        self.wait_timeout = wait_timeout
        self.window_size = window_size
        self.user_agent = user_agent

        # WebDriver components
        self.driver = None
        self.wait = None

        # Component instances
        self.gallery_selector = None
        self.gallery_crawler = None

        # Session state
        self.base_url = "https://charts.ecmwf.int/"
        self.session_data = {}

        self.setup_driver()

        logger.info("Crawler initialized with headless=%s, timeout=%ds", headless, wait_timeout)

    def setup_driver(self) -> bool:
        """
        Set up the Chrome WebDriver with optimized options.

        Returns:
            True if successful, False otherwise
        """
        if self.driver:
            logger.info("WebDriver already initialized")
            return True

        try:
            chrome_options = Options()

            # Basic options
            if self.headless:
                chrome_options.add_argument('--headless')

            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument(f'--window-size={self.window_size}')

            # Performance optimizations
            chrome_options.add_argument('--disable-extensions')
            chrome_options.add_argument('--disable-plugins')
            chrome_options.add_argument('--disable-images')  # Speed up loading
            chrome_options.add_argument('--disable-javascript')  # Can be enabled if needed

            # Privacy and security
            chrome_options.add_argument('--disable-web-security')
            chrome_options.add_argument('--disable-features=VizDisplayCompositor')

            # Custom user agent
            if self.user_agent:
                chrome_options.add_argument(f'--user-agent={self.user_agent}')

            # Initialize driver
            self.driver = webdriver.Chrome(options=chrome_options)
            self.wait = WebDriverWait(self.driver, self.wait_timeout)

            # Initialize component instances with shared driver
            self.gallery_selector = GallerySelector(
                driver=self.driver,
                headless=self.headless,
                wait_timeout=self.wait_timeout
            )
            self.gallery_crawler = GallaryCrawler(
                driver=self.driver,
                wait_timeout=self.wait_timeout
            )

            logger.info("WebDriver and components initialized successfully")
            return True

        except (WebDriverException, OSError) as e:
            logger.error("Failed to setup WebDriver: %s", e)
            return False

    def connect_to_gallery(self) -> bool:
        """
        Connect to the ECMWF Charts website.

        Returns:
            True if successful, False otherwise
        """
        if not self.driver and not self.setup_driver():
            return False

        try:
            success = self.gallery_crawler.navigate_to_gallery(self.base_url)

            if success:
                self.session_data['remote_url'] = self.base_url
                logger.info("Remote gallery connected successfully")

            return success

        except (TimeoutException, WebDriverException) as e:
            logger.error("Error connecting to remote gallery: %s", e)
            return False

    def apply_filters(self, filters: Dict[str, List[str]]) -> Dict[str, bool]:
        """
        Apply filters to the current gallery (local or remote).

        Args:
            filters: Dictionary with filter categories and values

        Returns:
            Dictionary with filter application results
        """
        if not self.gallery_selector:
            logger.error("Gallery selector not initialized")
            return {}

        try:
            results = self.gallery_selector.apply_filters(filters)
            logger.info("Applied %d filters with %d successes", 
                       len(results), sum(1 for r in results.values() if r))
            return results

        except (TimeoutException, WebDriverException) as e:
            logger.error("Error applying filters: %s", e)
            return {}

    def extract_gallery_metadata(self, max_items: int = 50) -> Dict[str, Any]:
        """
        Extract metadata from the current gallery.

        Args:
            max_items: Maximum number of items to extract

        Returns:
            Dictionary with extracted metadata
        """
        metadata = {
            'url': self.base_url,
            'timestamp': time.time(),
            'filters': {},
            'charts': [],
            'page_info': {}
        }

        try:
            if self.gallery_selector:
                metadata['filters'] = self.gallery_selector.get_all_available_filters()
                metadata['current_filter_states'] = self.gallery_selector.get_current_filter_states()

            elif self.gallery_crawler:
                if self.gallery_crawler:
                    metadata['page_info'] = self.gallery_crawler.get_page_info()
                    metadata['filters'] = self.gallery_crawler.extract_available_filters()
                    metadata['charts'] = self.gallery_crawler.extract_chart_metadata(self.base_url,max_items)
                    metadata['image_urls'] = self.gallery_crawler.get_chart_image_urls()

            logger.info("Extracted metadata for %s gallery", self.base_url)
            return metadata

        except (TimeoutException, WebDriverException) as e:
            logger.error("Error extracting metadata: %s", e)
            return metadata

    def get_current_status(self) -> Dict[str, Any]:
        """
        Get current crawler status and session information.

        Returns:
            Dictionary with status information
        """
        status = {
            'driver_active': self.driver is not None,
            'current_url': self.base_url,
            'components': {
                'gallery_selector': self.gallery_selector is not None,
                'gallery_crawler': self.gallery_crawler is not None
            },
            'session_data': self.session_data.copy()
        }

        if self.driver:
            try:
                status['browser_title'] = self.driver.title
                status['window_size'] = self.driver.get_window_size()
            except (TimeoutException, WebDriverException) as e:
                logger.debug("Error getting browser status: %s", e)

        return status

    def save_session_data(self, filename: str) -> bool:
        """
        Save current session data to a file.

        Args:
            filename: Output filename

        Returns:
            True if successful, False otherwise
        """
        try:
            session_info = {
                'status': self.get_current_status(),
                'metadata': self.extract_gallery_metadata(),
                'save_timestamp': time.time()
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(session_info, f, indent=2, ensure_ascii=False)

            logger.info("Session data saved to %s", filename)
            return True

        except (TimeoutException, WebDriverException) as e:
            logger.error("Error saving session data: %s", e)
            return False

    def cleanup(self) -> None:
        """Clean up resources and close WebDriver."""
        try:
            if self.gallery_selector:
                # GallerySelector cleanup is handled by shared driver
                pass

            if self.gallery_crawler:
                # GallaryCrawler cleanup is handled by shared driver
                pass

            if self.driver:
                self.driver.quit()
                self.driver = None
                self.wait = None
                logger.info("WebDriver closed and resources cleaned up")

            # Reset component references
            self.gallery_selector = None
            self.gallery_crawler = None
            self.session_data.clear()

        except (TimeoutException, WebDriverException) as e:
            logger.error("Error during cleanup: %s", e)

    def __enter__(self):
        """Context manager entry."""
        self.setup_driver()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


def main():
    """
    Example usage of the main Crawler class.
    """
    print("Weather Chart Gallery Crawler")
    print("=" * 40)

    # Example 1: Local HTML file processing
    print("\n1. Local HTML Gallery Processing")
    print("-" * 30)

    # Example 2: Remote website crawling
    print("\n2. Remote ECMWF Website Crawling")
    print("-" * 30)

    user_input = input("Connect to remote ECMWF website? (y/n): ").lower().strip()

    if user_input in ['y', 'yes']:
        with Crawler(headless=False) as crawler:
            success = crawler.connect_to_gallery()
            if success:
                print("✅ Connected to ECMWF Charts website")

                # Get page info
                status = crawler.get_current_status()
                print(f"✅ Page title: {status.get('browser_title', 'Unknown')}")

                # Extract metadata
                metadata = crawler.extract_gallery_metadata(max_items=5)
                charts_found = len(metadata.get('charts', []))
                print(f"✅ Found {charts_found} charts")

                # Save session data
                crawler.save_session_data('crawler_session.json')
                print("✅ Session data saved")

                input("Press Enter to continue...")
            else:
                print("❌ Failed to connect to remote gallery")
    else:
        print("⚠️ Skipping remote crawling")

    # Example 3: Mode switching
    print("\n3. Mode Switching Demo")
    print("-" * 20)

    with Crawler(headless=True) as crawler:
        # Start with remote
        if crawler.connect_to_gallery():
            print("✅ Started in remote mode")

        else:
            print("❌ Failed to start remote mode")

    print("\n🎉 Crawler demo completed!")


if __name__ == "__main__":
    main()
