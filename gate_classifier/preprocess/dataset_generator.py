"""
split raw dataset into training, validation and test datasets
"""
import random
import logging
from pathlib import Path
from pydantic import BaseModel
from PIL import Image
from ..setting import CURRENT_DATASET_DIR, RAW_CHART_DIR, RAW_WEATHER_DIR, GALLERY_DIR, DATASET_SIZE
from multi_label_classifier.preprocess import SplitConfig

logger = logging.getLogger(__name__)

class NonWeatherChartDatasetConfig(BaseModel):
    """
    config for the non weather chart dataset
    """
    dataset_dir: Path
    ratio: float

class DataSpliter:
    """
    split raw dataset into training, validation and test datasets
    """
    def __init__(self, split_config: SplitConfig):
        random.seed(split_config.seed)
        self.split_config = split_config
        self.non_weatherchart_dir_list = {
            "chart": NonWeatherChartDatasetConfig(dataset_dir=Path(RAW_CHART_DIR), ratio=0.4),
            "weather": NonWeatherChartDatasetConfig(dataset_dir=Path(RAW_WEATHER_DIR), ratio=0.6)
        }
        self.weather_dir = Path(GALLERY_DIR)
        self.output_dir = Path(CURRENT_DATASET_DIR)
        self.dataset_size = DATASET_SIZE * 2

    def split(self):
        """
        split the data into train, validation and test sets and call Migrate to build the data batch
        """
        train_indices = range(int(self.dataset_size * self.split_config.train_ratio))
        validation_indices = range(int(self.dataset_size * self.split_config.validation_ratio))
        test_indices = range(self.dataset_size - len(train_indices) - len(validation_indices))
        random.shuffle(train_indices)
        random.shuffle(validation_indices)
        random.shuffle(test_indices)
        non_weatherchart_indices = {
            "train": train_indices[:self.dataset_size//2],
            "validation": validation_indices[:self.dataset_size//2],
            "test": test_indices[:self.dataset_size//2]
        }
        for dataset_name, dataset_indices in non_weatherchart_indices.items():
            length_of_dataset = {
                "chart": int(len(dataset_indices) * self.non_weatherchart_dir_list["chart"].ratio),
                "weather": len(dataset_indices) - int(len(dataset_indices) * self.non_weatherchart_dir_list["chart"].ratio)
            }
            for source_name, source in self.non_weatherchart_dir_list.items():
                files = random.choices(list(source.dataset_dir.glob("*.png")), k=length_of_dataset[source_name])
                for i, file in enumerate(files):
                    image = Image.open(file)
                    image = image.convert('RGB')
                    image.save(self.output_dir / dataset_name / f"{dataset_indices[i]:06d}.png")

        weather_indices = {
            "train": train_indices[self.dataset_size//2:],
            "validation": validation_indices[self.dataset_size//2:],
            "test": test_indices[self.dataset_size//2:]
        }
        for dataset_name, dataset_indices in weather_indices.items():
            files = random.choices(list(self.weather_dir.glob("*.png")), k=len(dataset_indices))
            for i, file in enumerate(files):
                image = Image.open(file)
                image = image.convert('RGB')
                image.save(self.output_dir / dataset_name / f"{dataset_indices[i]:06d}.png")

if __name__ == "__main__":
    spliter = DataSpliter(SplitConfig())
    spliter.split()
