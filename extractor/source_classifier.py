"""
Classify the source of the image
"""

import logging
import re
import hashlib
import os
import json
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class RegexPattern(BaseModel):
    """
    Regex pattern class
    """
    cma_pattern : re.Pattern = re.compile(r"^(\d{8})-早会商-信息中心-实况$") # 国家气象信息中心 <YYYYMMDD-早会商-信息中心-实况>
    aoc_pattern : re.Pattern = re.compile(r"^aoc(\d{8})$") # 国家气象局气象探测中心 <aocYYYYMMDD>
    nmc_pattern : re.Pattern = re.compile(r"^(\d{4})年(\d{2})月(\d{2})日早间会商首席发言$") # 中央气象台 <YYYY年MM月DD日早间会商首席发言>

class ImageInfo(BaseModel):
    """
    Image information class
    """
    file_path : str
    ppt_name : str
    slide_index : int
    img_index : int
    format : str
    
    class Config:
        frozen = True  # make object immutable to ensure this object hashable

class SourceClassifier:
    """
    Image source classifier class
    """
    def __init__(self, image_dir : str):
        self.regex_pattern = RegexPattern()
        self.root_dir = image_dir
        self.cma_image_list : dict[ImageInfo, str] = {}
        self.aoc_image_list : dict[ImageInfo, str] = {}

    def classify_source(self, image_path : str) -> str:
        """
        Classify the source of the image
        """
        image_info : ImageInfo = self.extract_image_info(image_path)
        if self.regex_pattern.cma_pattern.match(image_info.ppt_name):
            self.cma_image_list[image_info] = self.compute_hashes(image_path)
            return "cma"
        if self.regex_pattern.aoc_pattern.match(image_info.ppt_name):
            self.aoc_image_list[image_info] = self.compute_hashes(image_path)
            return "aoc"
        if self.regex_pattern.nmc_pattern.match(image_info.ppt_name):
            return "nmc"
        return "unknown"

    def extract_image_info(self, image_path : str) -> ImageInfo:
        """
        Extract the information of the image
        """
        file_name = os.path.basename(image_path)
        return ImageInfo(
            file_path=os.path.abspath(image_path),
            ppt_name=file_name.split("_")[0],
            slide_index=int(file_name.split("_")[1].split("slide")[1].split(".")[0]),
            img_index=int(file_name.split("_")[2].split("img")[1].split(".")[0]),
            format=file_name.split(".")[1]
        )

    def compute_hashes(self, image_path : str) -> str:
        """
        Compute the sha256 hash of the image
        """
        with open(image_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def check_nmc_image(self, image_list : list[str]) -> None:
        """
        Check the image is from the nmc
        """
        for image_path in image_list:
            image_hash = self.compute_hashes(image_path)
            if image_hash in self.aoc_image_list.values():
                image_info : ImageInfo = self.extract_image_info(image_path)
                logger.debug("Image %s is from aoc", os.path.abspath(image_path))
                self.aoc_image_list[image_info] = image_hash
            if image_hash in self.cma_image_list.values():
                image_info : ImageInfo = self.extract_image_info(image_path)
                logger.debug("Image %s is from cma", os.path.abspath(image_path))
                self.cma_image_list[image_info] = image_hash

    def save_classified_list(self, file_name : str) -> None:
        """
        save the classified list as a json file
        """
        try:
            with open(file_name, "w", encoding="utf-8") as f:
                # convert ImageInfo object to a serializable dictionary
                data = [
                    {**info.model_dump(), "hash": hash_val, "type": "cma"}
                    for info, hash_val in self.cma_image_list.items()
                ]
                data.extend([
                    {**info.model_dump(), "hash": hash_val, "type": "aoc"}
                    for info, hash_val in self.aoc_image_list.items()
                ])
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Failed to save the classified list: %s", e)
            raise e
