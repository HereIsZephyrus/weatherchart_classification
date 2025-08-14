"""
This module is used to count the frequency of each parameter in the gallery.
"""
from typing import Dict, List, Set
import logging

logger = logging.getLogger(__name__)

def reorganize_gallery(gallery: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Reorganize gallery data to group URLs that appear in multiple parameters.

    Args:
        gallery: Original gallery dictionary, keys are parameter names, values are URL lists

    Returns:
        Reorganized gallery dictionary, ensuring each URL belongs to only one type
    """
    # standardize the parameter names first
    standardized_gallery = {}
    for param in gallery.keys():
        new_param = param.lower()
        new_param = new_param.replace(" ", "_")
        new_param = new_param.replace("-", "_")
        standardized_gallery[new_param] = gallery[param]
    gallery = standardized_gallery
    logger.info("standardized gallery")

    url_to_params: Dict[str, Set[str]] = {}

    # Count how many times each URL appears in which parameters
    for param, urls in gallery.items():
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

    logger.info("reorganized gallery")
    return new_gallery
