"""
Dataset generate module
"""

import json
import random
import logging
from pathlib import Path
from typing import List, Dict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel
from .chart import Chart, ChartMetadata
from .chart_enhancer import ChartEnhancer, EnhancerConfig
from ..constants import DATASET_DIR, GALLERY_DIR, RADER_RAW_DIR

logger = logging.getLogger(__name__)

class DataBatchRole(tuple[str, float], Enum):
    """
    role of the data batch.
    the first element is the role name,
    the second element is the type percentage that contained in one data batch.
    """
    TRAIN = ("train", 0.7)
    VALIDATION = ("validation", 1.0)
    TEST = ("test", 0.9)

class DataBatchStatus(str, Enum):
    """
    status of the data batch
    """
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"

class DataBatchMetedata(BaseModel):
    """
    metadata for a batch of data
    """
    batch_id: int
    name: str
    path: str
    size: int
    progress: float
    status: DataBatchStatus
    role: DataBatchRole

class DatasetConfig(BaseModel):
    """
    config for the dataset
    """
    batch_num: int
    single_batch_size: int
    train_percent : float = 0.7
    validation_percent : float = 0.1
    test_percent : float = 0.2

class DataBatchBuilder:
    """
    builder for the data batch
    """
    def __init__(self, size : int, bid : int, role: DataBatchRole, enhancer_config: EnhancerConfig):
        self.metadata = self.construct_metadata(bid, size, role)
        self.enhancer = ChartEnhancer(enhancer_config)

    def construct_metadata(self, bid : int, size: int, role: DataBatchRole) -> DataBatchMetedata:
        """
        construct the metadata for a batch of data
        """
        name = f"{role.value[0]}_batch_{bid:04d}"
        return DataBatchMetedata(
            batch_id=bid,
            name=name,
            path=f"{DATASET_DIR}/{name}",
            size=size,
            progress=0,
            status=DataBatchStatus.PENDING,
            role=role
        )

    def collect_folder(self, folder_path: str = "") -> List[str]:
        """
        collect all images from the gallery directory
        """
        image_files = []

        gallery_path = Path(folder_path)
        if gallery_path.exists():
            for img_file in gallery_path.glob("*.webp"):
                image_files.append(str(img_file))
        else:
            logger.warning("No gallery path found")

        return image_files

    def boostraping(self, sample_num: int, folder_path: str = "") -> List[str]:
        """
        boostraping sample list
        """
        all_images = self.collect_folder(folder_path)
        if not all_images:
            logger.warning("No image files found")
            return []

        # Bootstrapping
        sampled_images = random.choices(all_images, k=sample_num)
        logger.info("Bootstrap sampling generated %d samples", len(sampled_images))
        return sampled_images

    def select_type(self) -> List[str]:
        """
        select the type of the data batch
        """
        percentage = self.metadata.role.value[1]
        size = int(self.metadata.size * percentage)
        selected_types = random.choices(GALLERY_DIR, k=size)
        return selected_types

    def generate_image_dataset(self, image_path_list: List[str], save_dir: Path) -> List[ChartMetadata]:
        """
        generate the data for a data batch
        """
        metadata_list : List[ChartMetadata] = []
        for index, image_path in enumerate(image_path_list):
            chart = self.enhancer(Chart(image_path, index))
            chart.save(save_dir / f"{index:04d}.png")
            metadata_list.append(chart.metadata)

        return metadata_list

    def sample_huggingface_dataset(self, sample_num: int, save_dir: Path, source_dir: str) -> List[ChartMetadata]:
        """
        generate the radar data for a data batch
        """
        metadata_list : List[ChartMetadata] = []
        return metadata_list

    def build(self) -> None:
        """
        main function to build a data batch
        """
        batch_dir = Path(self.metadata.path)
        batch_dir.mkdir(parents=True, exist_ok=True)

        images_dir = batch_dir / "images"
        labels_dir = batch_dir / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)

        types = self.select_type()
        image_num_each_type = int(self.metadata.size / (len(types) + 1)) # consider radar data

        sampled_images = []
        for type_folder in types:
            type_images = self.boostraping(sample_num=image_num_each_type, folder_path=type_folder)
            if not type_images:
                logger.warning("No image files found in %s", type_folder)
                continue
            sampled_images.extend(type_images)
        ecmwf_labels = self.generate_image_dataset(
            image_path_list=sampled_images,
            save_dir=images_dir
        )

        radar_num = self.metadata.size - len(sampled_images)
        radar_labels = self.sample_huggingface_dataset(
            sample_num=radar_num,
            save_dir=images_dir,
            source_dir=RADER_RAW_DIR
        )

        total_labels = ecmwf_labels + radar_labels
        with open(labels_dir / "labels.json", "w", encoding="utf-8") as f:
            json.dump(total_labels, f, ensure_ascii=False, indent=2)

        logger.info("Start generating batch data %s", self.metadata.batch_id)

class DatasetManager:
    """
    manager for the dataset
    """
    def __init__(self, config: DatasetConfig, enhancer_config: EnhancerConfig):
        self.dataset_dir = Path(DATASET_DIR)
        self.config = config
        self.batch_metadata_list: List[DataBatchMetedata] = []
        self.batch_list: List[str] = []
        self.enhancer = ChartEnhancer(enhancer_config)
        self.standardize_percentage()

    def standardize_percentage(self) -> None:
        """
        ensure the sum of the percentage of train, validation and test is 1
        """
        total = self.config.train_percent + self.config.validation_percent + self.config.test_percent

        if abs(total - 1.0) > 1e-6:
            logger.warning("The sum of the percentage is not 1 (%.3f), automatically standardize", total)

            self.config.train_percent /= total
            self.config.validation_percent /= total
            self.config.test_percent /= total

            logger.info("After standardization, the ratio - train: %.2f, validation: %.2f, test: %.2f",
                       self.config.train_percent,
                       self.config.validation_percent,
                       self.config.test_percent)

    def generate_dataset_build_task(self) -> None:
        """
        生成数据集构建任务
        """
        self.batch_metadata_list.clear()
        self.batch_list.clear()

        # 计算各个数据集的批次数量
        train_batches = int(self.config.batch_num * self.config.train_percent)
        val_batches = int(self.config.batch_num * self.config.validation_percent)
        test_batches = self.config.batch_num - train_batches - val_batches  # 确保总数正确

        logger.info("生成数据集任务 - 训练批次: %d, 验证批次: %d, 测试批次: %d",
                   train_batches, val_batches, test_batches)

        # 生成训练批次
        for i in range(train_batches):
            batch_name = f"train_batch_{i:04d}"
            metadata = construct_metadata(
                str(self.dataset_dir),
                batch_name,
                self.config.single_batch_size,
                DataBatchRole.TRAIN
            )
            self.batch_metadata_list.append(metadata)
            self.batch_list.append(batch_name)

        # 生成验证批次
        for i in range(val_batches):
            batch_name = f"val_batch_{i:04d}"
            metadata = construct_metadata(
                str(self.dataset_dir),
                batch_name,
                self.config.single_batch_size,
                DataBatchRole.VALIDATION
            )
            self.batch_metadata_list.append(metadata)
            self.batch_list.append(batch_name)

        # 生成测试批次
        for i in range(test_batches):
            batch_name = f"test_batch_{i:04d}"
            metadata = construct_metadata(
                str(self.dataset_dir),
                batch_name,
                self.config.single_batch_size,
                DataBatchRole.TEST
            )
            self.batch_metadata_list.append(metadata)
            self.batch_list.append(batch_name)

        logger.info("生成了 %d 个批次任务", len(self.batch_metadata_list))

    def _process_single_batch(self, metadata: DataBatchMetedata) -> Dict:
        """
        处理单个批次的内部方法
        """
        try:
            # 更新状态为处理中
            metadata.status = DataBatchStatus.PROCESSING

            # 生成批次数据
            success_count = generate_data_batch(metadata, self.enhancer, self.source_images)

            # 更新状态为完成
            metadata.status = DataBatchStatus.COMPLETED
            metadata.progress = 1.0

            return {
                "batch_id": metadata.id,
                "status": "success",
                "actual_size": success_count,
                "target_size": metadata.size
            }

        except (IOError, ValueError, OSError) as e:
            logger.error("批次 %s 处理失败: %s", metadata.id, e)
            metadata.status = DataBatchStatus.PENDING
            metadata.progress = 0.0

            return {
                "batch_id": metadata.id,
                "status": "failed",
                "error": str(e),
                "actual_size": 0,
                "target_size": metadata.size
            }

    def build_dataset(self, max_workers: int = 4) -> Dict:
        """
        并行构建数据集
        """
        if not self.batch_metadata_list:
            logger.error("没有找到批次任务，请先调用 generate_dataset_build_task()")
            return {"status": "error", "message": "No batch tasks found"}

        # 确保输出目录存在
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        logger.info("开始并行构建数据集，使用 %d 个工作线程", max_workers)

        results = []
        completed_batches = 0
        failed_batches = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_metadata = {
                executor.submit(self._process_single_batch, metadata): metadata
                for metadata in self.batch_metadata_list
            }

            # 处理完成的任务
            for future in as_completed(future_to_metadata):
                metadata = future_to_metadata[future]
                try:
                    result = future.result()
                    results.append(result)

                    if result["status"] == "success":
                        completed_batches += 1
                        logger.info("批次 %s 完成 (%d/%d)",
                                  result["batch_id"],
                                  completed_batches + failed_batches,
                                  len(self.batch_metadata_list))
                    else:
                        failed_batches += 1
                        logger.error("批次 %s 失败", result["batch_id"])

                except (IOError, ValueError, OSError) as e:
                    logger.error("批次 %s 处理异常: %s", metadata.id, e)
                    failed_batches += 1
                    results.append({
                        "batch_id": metadata.id,
                        "status": "exception",
                        "error": str(e),
                        "actual_size": 0,
                        "target_size": metadata.size
                    })

        # 保存数据集信息
        self._save_dataset_info(results)

        # 生成汇总报告
        summary = {
            "status": "completed",
            "total_batches": len(self.batch_metadata_list),
            "completed_batches": completed_batches,
            "failed_batches": failed_batches,
            "success_rate": completed_batches / len(self.batch_metadata_list),
            "total_images": sum(r.get("actual_size", 0) for r in results),
            "target_images": sum(r.get("target_size", 0) for r in results),
            "results": results
        }

        logger.info("数据集构建完成 - 成功: %d/%d 批次, 总图像: %d",
                   completed_batches, len(self.batch_metadata_list), summary["total_images"])

        return summary

    def _save_dataset_info(self, results: List[Dict]) -> None:
        """
        保存数据集信息到文件
        """
        dataset_info = {
            "config": self.config.model_dump(),
            "enhancer_config": self.enhancer.config.model_dump(),
            "batch_metadata": [metadata.model_dump() for metadata in self.batch_metadata_list],
            "build_results": results,
            "summary": {
                "total_batches": len(self.batch_metadata_list),
                "total_target_images": sum(metadata.size for metadata in self.batch_metadata_list),
                "total_actual_images": sum(r.get("actual_size", 0) for r in results),
                "train_batches": len([m for m in self.batch_metadata_list if m.role == DataBatchRole.TRAIN]),
                "val_batches": len([m for m in self.batch_metadata_list if m.role == DataBatchRole.VALIDATION]),
                "test_batches": len([m for m in self.batch_metadata_list if m.role == DataBatchRole.TEST])
            }
        }

        info_path = self.dataset_dir / "dataset_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)

        logger.info("数据集信息已保存到: %s", info_path)

    def get_dataset_statistics(self) -> Dict:
        """
        获取数据集统计信息
        """
        if not self.dataset_dir.exists():
            return {"error": "Dataset directory not found"}

        stats = {
            "total_batches": len(self.batch_metadata_list),
            "roles": {},
            "status": {},
            "total_images": 0
        }

        for metadata in self.batch_metadata_list:
            # 统计角色分布
            role = metadata.role.value
            if role not in stats["roles"]:
                stats["roles"][role] = 0
            stats["roles"][role] += 1

            # 统计状态分布
            status = metadata.status.value
            if status not in stats["status"]:
                stats["status"][status] = 0
            stats["status"][status] += 1

            # 统计图像数量
            if metadata.status == DataBatchStatus.COMPLETED:
                batch_dir = Path(metadata.path)
                images_dir = batch_dir / "images"
                if images_dir.exists():
                    stats["total_images"] += len(list(images_dir.glob("*.jpg")))

        return stats
