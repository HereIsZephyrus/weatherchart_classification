"""
Abstract chart class from Pillow image and dataset metadata
"""

import json
import os
from typing import List, Optional
import logging
import random
from datetime import datetime
from PIL import Image
from pydantic import BaseModel
from ..constants import GALLERY_MAPPING_BILINGUAL_PATH

logger = logging.getLogger(__name__)

class ChartMetadata(BaseModel):
    """
    Metadata for a chart
    """
    index: int
    en_name: str
    zh_name: str
    time: datetime
    label: List[str]

class Chart:
    """
    Abstract chart class from Pillow image and dataset metadata
    """
    name_mapping : Optional[dict[str, tuple[str, str]]] = None

    def __init__(self, image_path: str, index: int = None):
        self.image_path = image_path
        self.image : Image.Image = Image.open(image_path)
        self.metadata = None
        self.load_metadata(index)

    def __del__(self):
        self.image.close()

    def __str__(self):
        return f"{self.metadata.en_name}({self.metadata.zh_name}): publish time<{self.metadata.time}>, label<{self.metadata.label}>. {self.image_path}"

    @property
    def metadata(self) -> ChartMetadata:
        """
        Get the metadata of the chart
        """
        return self.metadata

    def load_name_mapping(self):
        """
        Load name mapping from gallery_mapping_bilingual.json
        """
        with open(GALLERY_MAPPING_BILINGUAL_PATH, "r", encoding="utf-8") as f:
            self.name_mapping = json.load(f)

    def load_metadata(self, index: int):
        """
        Load metadata from image_path info
        """
        self.metadata.label = []
        self.metadata.index = index

        if self.name_mapping is None:
            self.load_name_mapping()

        basename = os.path.basename(self.image_path)
        elements = basename.split("_")[1]
        self.metadata.time = datetime.strptime(elements, "%Y%m%d%H%M")
        self.metadata.en_name, self.metadata.zh_name = self.name_mapping[basename]

    def construct_title(self) -> str:
        """
        Construct the title of the chart
        """
        year = random.randint(2010, 2030)
        month = random.randint(1, 12)
        day = random.randint(1, 30)
        if month == 2 and day > 28:
            day = 28
        return f"{year}年{month}月{day}日{self.metadata.zh_name}图"

    def save(self, save_path: str):
        """
        Save the chart to the save_path
        """
        self.image.save(save_path)
