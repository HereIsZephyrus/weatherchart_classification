"""
Dataset generate module
"""

import json
import random
import logging
from pathlib import Path
from typing import List, Dict
from enum import Enum
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

from .chart_enhancer import ChartEnhancer, EnhancerConfig
from ..constants import DATASET_DIR, GALLERY_DIR, RADER_RAW_DIR, IMAGE_DIR

class DataBatchRole(str, Enum):
    """
    role of the data batch
    """
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

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
    id: str
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

def construct_metadata(dataset_dir: str, name : str, size: int, role: DataBatchRole):
    """
    construct the metadata for a batch of data
    """
    return DataBatchMetedata(
        id=name,
        path=f"{dataset_dir}/{name}",
        size=size,
        progress=0,
        status=DataBatchStatus.PENDING,
        role=role
    )

logger = logging.getLogger(__name__)

def collect_all_images() -> List[str]:
    """
    从TRAIN_DATA_DIR中收集所有可用的图像文件
    """
    image_files = []
    
    # 收集gallery中的图像
    gallery_path = Path(GALLERY_DIR)
    if gallery_path.exists():
        for subdir in gallery_path.iterdir():
            if subdir.is_dir():
                for img_file in subdir.glob("*.webp"):
                    image_files.append(str(img_file))
                for img_file in subdir.glob("*.jpg"):
                    image_files.append(str(img_file))
                for img_file in subdir.glob("*.png"):
                    image_files.append(str(img_file))
    
    # 收集radar-dataset中的图像
    radar_path = Path(RADER_RAW_DIR)
    if radar_path.exists():
        # 查找images目录
        images_dir = radar_path / "images"
        if images_dir.exists():
            for img_file in images_dir.glob("*.jpg"):
                image_files.append(str(img_file))
            for img_file in images_dir.glob("*.png"):
                image_files.append(str(img_file))
    
    # 收集extracted_images中的图像
    extracted_path = Path(IMAGE_DIR)
    if extracted_path.exists():
        for img_file in extracted_path.glob("*.jpg"):
            image_files.append(str(img_file))
        for img_file in extracted_path.glob("*.png"):
            image_files.append(str(img_file))
    
    logger.info("收集到 %d 个图像文件", len(image_files))
    return image_files

def boostraping_sample_list(sample_num: int) -> List[str]:
    """
    boostraping采样样本列表
    """
    all_images = collect_all_images()
    if not all_images:
        logger.warning("没有找到任何图像文件")
        return []
    
    # Bootstrap采样：有放回抽样
    sampled_images = random.choices(all_images, k=sample_num)
    logger.info("Bootstrap采样生成 %d 个样本", len(sampled_images))
    return sampled_images

def generate_data_batch(metadata: DataBatchMetedata, enhancer: ChartEnhancer, source_images: List[str]):
    """
    生成一个数据批次
    """
    batch_dir = Path(metadata.path)
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = batch_dir / "images"
    labels_dir = batch_dir / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    
    logger.info("开始生成批次 %s，目标大小: %d", metadata.id, metadata.size)
    
    # 从源图像中随机选择并增强
    if len(source_images) < metadata.size:
        # 如果源图像不足，使用bootstrap采样
        selected_images = boostraping_sample_list(metadata.size)
    else:
        selected_images = random.sample(source_images, metadata.size)
    
    success_count = 0
    for i, img_path in enumerate(selected_images):
        try:
            # 加载图像
            with Image.open(img_path) as image:
                # 使用ChartEnhancer进行增强
                enhanced_image = enhancer(image)
                
                # 生成文件名
                filename = f"{metadata.id}_{i:06d}"
                
                # 保存增强后的图像
                image_path = images_dir / f"{filename}.jpg"
                enhanced_image.save(image_path, 'JPEG', quality=95)
                
                # 生成标签信息
                label_info = {
                    "image_id": filename,
                    "source_path": img_path,
                    "batch_id": metadata.id,
                    "batch_role": metadata.role.value,
                    "enhanced": True,
                    "image_size": enhanced_image.size
                }
                
                # 保存标签文件
                label_path = labels_dir / f"{filename}.json"
                with open(label_path, 'w', encoding='utf-8') as f:
                    json.dump(label_info, f, ensure_ascii=False, indent=2)
                
                success_count += 1
                
                # 更新进度
                progress = (i + 1) / len(selected_images)
                if i % 100 == 0 or i == len(selected_images) - 1:
                    logger.info("批次 %s 进度: %d/%d (%.1f%%)", 
                              metadata.id, i + 1, len(selected_images), progress * 100)
                              
        except (IOError, ValueError, OSError) as e:
            logger.error("处理图像 %s 失败: %s", img_path, e)
            continue
    
    # 创建批次信息文件
    batch_info = {
        "batch_id": metadata.id,
        "role": metadata.role.value,
        "target_size": metadata.size,
        "actual_size": success_count,
        "source_images_count": len(selected_images),
        "success_rate": success_count / len(selected_images) if selected_images else 0
    }
    
    info_path = batch_dir / "batch_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(batch_info, f, ensure_ascii=False, indent=2)
    
    logger.info("批次 %s 生成完成: %d/%d 成功", metadata.id, success_count, len(selected_images))
    return success_count

class DatasetManager:
    """
    manager for the dataset
    """
    def __init__(self, config: DatasetConfig, enhancer_config: EnhancerConfig = None):
        self.dataset_dir = Path(DATASET_DIR)
        self.config = config
        self.batch_metadata_list: List[DataBatchMetedata] = []
        self.batch_list: List[str] = []
        self.source_images: List[str] = []
        
        # 创建增强器
        if enhancer_config is None:
            enhancer_config = EnhancerConfig(
                use_clip=True,
                add_logo_prob=0.3,
                add_title_prob=0.4,
                clip_chart_prob=0.2,
                hue_shift_prob=0.3,
                contrast_prob=0.4,
                brightness_prob=0.3,
                saturation_prob=0.3
            )
        self.enhancer = ChartEnhancer(enhancer_config)
        
        # 标准化百分比
        self.standardize_percentage()
        
        # 初始化时收集源图像
        self.source_images = collect_all_images()
        logger.info("DatasetManager初始化完成，找到 %d 个源图像", len(self.source_images))

    def standardize_percentage(self) -> None:
        """
        确保训练、验证、测试集的百分比之和为1
        """
        total = self.config.train_percent + self.config.validation_percent + self.config.test_percent
        
        if abs(total - 1.0) > 1e-6:  # 允许浮点误差
            logger.warning("百分比之和不为1 (%.3f)，自动标准化", total)
            
            # 标准化
            self.config.train_percent /= total
            self.config.validation_percent /= total
            self.config.test_percent /= total
            
            logger.info("标准化后比例 - 训练: %.2f, 验证: %.2f, 测试: %.2f", 
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
