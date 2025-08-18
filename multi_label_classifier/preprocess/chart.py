"""
Abstract chart class from Pillow image and dataset metadata
"""

import json
import os
import re
import ast
from typing import List, Optional, Any
import logging
import random
from PIL import Image
from pydantic import BaseModel
from pandas import Series
from ...constants import GALLERY_MAPPING_BILINGUAL_PATH
logger = logging.getLogger(__name__)

class ChartMetadata(BaseModel):
    """
    Metadata for a chart
    """
    index: int
    en_name: str
    zh_name: Optional[str] = None
    label: List[str]
    summary: Optional[str] = None

class Chart:
    """
    Abstract chart class from Pillow image and dataset metadata
    """
    name_mapping : Optional[dict[str, tuple[str, str]]] = None

    def __init__(self, image_path: str, index: int, info: Series = None):
        self.image_path = image_path
        self.image : Image.Image = Image.open(image_path)
        self._metadata = None
        if info is None:
            self.load_metadata(index)
        else:
            self.load_metadata_from_info(index, info)

    def __del__(self):
        if hasattr(self, 'image') and self.image is not None:
            self.image.close()

    def __str__(self):
        return f"{self._metadata.en_name}({self._metadata.zh_name}): label<{self._metadata.label}>. {self.image_path}"

    @property
    def metadata(self) -> ChartMetadata:
        """
        Get the metadata of the chart
        """
        result = self._metadata.model_dump()
        return result

    def __getitem__(self, key: str) -> Any:
        """
        Get the metadata of the chart
        """
        return self._metadata.model_dump()[key]

    def load_name_mapping(self):
        """
        Load name mapping from gallery_mapping_bilingual.json
        """
        with open(GALLERY_MAPPING_BILINGUAL_PATH, "r", encoding="utf-8") as f:
            self.name_mapping = json.load(f)

    def load_metadata_from_info(self, index: int, info: Series):
        """
        Load metadata from info
        """
        self._metadata = ChartMetadata(
            index=index,
            en_name=info.en,
            label=ast.literal_eval(info.feature) if isinstance(info.feature, str) else (list(info.feature) if hasattr(info.feature, '__iter__') else [info.feature]),
            summary=info.summary,
        )

    def load_metadata(self, index: int):
        """
        Load metadata from image_path info
        """
        if self.name_mapping is None:
            self.load_name_mapping()

        basename = os.path.basename(self.image_path).split(".")[0]
        time = re.search(r"\d{12}", basename).group(0) # YYYYMMDDHHMM
        product_name = basename.split(time)[0][:-1] # the last char is "_"

        # Create new metadata instance
        self._metadata = ChartMetadata(
            index=index,
            en_name=self.name_mapping[product_name]['en'],
            zh_name=self.name_mapping[product_name]['zh'],
            label=os.path.abspath(self.image_path).split("/")[-2].split("A"),
            summary=""
        )

    def construct_title(self) -> str:
        """
        Construct the title of the chart
        """
        year = random.randint(2010, 2030)
        month = random.randint(1, 12)
        day = random.randint(1, 30)
        if month == 2 and day > 28:
            day = 28
        return f"{month}{day}{self._metadata.zh_name}图".encode('utf-8').decode('utf-8')

    def save(self, save_path: str):
        """
        Save the chart to the save_path
        """
        # convert to RGB and save
        self.image = self.image.convert('RGB')
        self.image.save(save_path)
