"""
parse hugging face dataset @deepguess--weather-analysis-dataset 
Convert to image-label format and save to local TRAIN_DATA_DIR/RADAR_DIR
"""

import os
import json
import logging
from typing import Dict, Any
from datasets import load_from_disk, load_dataset
from PIL import Image
from ..constants import RADAR_DIR, RADAR_RAW_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RadarDatasetParser:
    """
    @deepguess--weather-analysis-dataset parser
    """

    def __init__(self):
        self.dataset = None
        self.output_dir = os.path.abspath(RADAR_DIR)
        self.images_dir = os.path.join(self.output_dir, "images")
        self.labels_dir = os.path.join(self.output_dir, "labels")

    def setup_directories(self):
        """
        create necessary directory structure
        """
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        logger.info("directory structure created: %s", self.output_dir)

    def load_dataset(self, dataset_path: str):
        """
        load dataset from hugging face
        """
        try:
            logger.info("loading dataset: %s", dataset_path)
            self.dataset = load_dataset(
                "deepguess/weather-analysis-dataset",
                cache_dir=os.path.abspath(dataset_path),
                download_mode="reuse_cache_if_exists",
            )
            logger.info("dataset loaded successfully, %d samples", len(self.dataset))
            return True
        except Exception as e:
            logger.error("dataset loading failed: %s", e)
            return False

    def create_label_dict(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        create label dictionary for each sample
        """
        label = {
            # basic information
            "image_id": example.get("image_id", ""),
            "caption": example.get("caption", ""),
            "product_type": example.get("product_type", ""),
            "meteorological_significance": example.get("meteorological_significance", ""),
            "context_summary": example.get("context_summary", ""),

            "parameters_visible": example.get("parameters_visible", "").split("|") if example.get("parameters_visible") else [],
            "key_features": example.get("key_features", "").split("|") if example.get("key_features") else [],

            "qa_pairs": [],
            "educational_content": {}
        }

        # add qa pairs
        for i in range(1, 4):
            question_key = f"qa_{i}_question"
            answer_key = f"qa_{i}_answer"
            difficulty_key = f"qa_{i}_difficulty"

            if example.get(question_key) and example.get(answer_key):
                label["qa_pairs"].append({
                    "question": example[question_key],
                    "answer": example[answer_key],
                    "difficulty": example.get(difficulty_key, "unknown")
                })

        # add educational content
        if example.get("edu_beginner_question"):
            label["educational_content"] = {
                "question": example.get("edu_beginner_question", ""),
                "options": example.get("edu_beginner_options", "").split("|") if example.get("edu_beginner_options") else [],
                "correct_answer_index": example.get("edu_beginner_correct", -1),
                "explanation": example.get("edu_beginner_explanation", "")
            }

        return label

    def save_image_and_label(self, example: Dict[str, Any], index: int):
        """
        save image and corresponding label file
        """
        try:
            # get image id as file name, if not, use index
            image_id = example.get("image_id", f"sample_{index:06d}")

            # save image
            image = example["image"]
            if isinstance(image, str):
                # if it is a path string, read image
                image = Image.open(image)

            image_filename = f"{image_id}.jpg"
            image_path = self.images_dir / image_filename

            # convert to RGB format (if needed) and save
            if image.mode in ('RGBA', 'LA'):
                image = image.convert('RGB')
            image.save(image_path, 'JPEG', quality=95)

            # create and save label
            label = self.create_label_dict(example)
            label_filename = f"{image_id}.json"
            label_path = self.labels_dir / label_filename

            with open(label_path, 'w', encoding='utf-8') as f:
                json.dump(label, f, ensure_ascii=False, indent=2)

            logger.info("saved: %s and %s", image_filename, label_filename)
            return True

        except Exception as e:
            logger.error("error saving %d-th sample: %s", index, e)
            return False

    def create_dataset_info(self):
        """
        create dataset info file
        """
        info = {
            "dataset_name": "deepguess/weather-analysis-dataset",
            "total_samples": len(self.dataset),
            "description": "天气雷达和卫星图像数据集，包含专家分析和教育内容",
            "structure": {
                "images": "images/ 目录包含所有图像文件 (.jpg格式)",
                "labels": "labels/ 目录包含对应的标签文件 (.json格式)",
                "filename_pattern": "使用image_id作为文件名，如果缺失则使用sample_XXXXXX"
            },
            "label_format": {
                "image_id": "图像唯一标识符",
                "caption": "气象专家描述",
                "product_type": "天气产品类型",
                "meteorological_significance": "气象学意义",
                "context_summary": "背景环境总结",
                "parameters_visible": "可见参数列表",
                "key_features": "关键特征列表",
                "qa_pairs": "问答对列表，包含问题、答案和难度",
                "educational_content": "教育内容，包含选择题、选项、正确答案和解释"
            }
        }

        info_path = os.path.join(self.output_dir, "dataset_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        logger.info("dataset info saved to: %s", info_path)

    def convert_dataset(self):
        """
        execute complete dataset conversion process
        """
        logger.info("starting dataset conversion...")

        # set directory
        self.setup_directories()

        # load dataset
        if not self.load_dataset(RADAR_RAW_DIR):
            return False

        # convert each sample
        success_count = 0
        for index, example in enumerate(self.dataset):
            if self.save_image_and_label(example, index):
                success_count += 1

            if (index + 1) % 100 == 0:
                logger.info("processed %d/%d samples", index + 1, len(self.dataset))

        # create dataset info file
        self.create_dataset_info()

        logger.info("conversion completed! successfully processed %d/%d samples", success_count, len(self.dataset))
        logger.info("data saved to: %s", self.output_dir)

        return success_count == len(self.dataset)
