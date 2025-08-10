├── gallery_inspector.py      # 数据集检查与统计（覆盖率、缺失项、投影分布）

- 统计下载数据集（覆盖率、缺失项等）：

```python
from crawler import GalleryInspector

inspector = GalleryInspector(base_dir="gallery")
stats = inspector.inspect()     # 计算统计
inspector.gallery_info()        # 打印简要汇总到日志

# 如需自定义期望投影集合：
# inspector = GalleryInspector(base_dir="gallery", projections=[
#     "opencharts_eastern_asia",
#     "opencharts_eruasia",
#     "opencharts_south_east_asia_and_indonesia",
#     "opencharts_southern_asia",
# ])
# inspector.inspect(); inspector.gallery_info()
```
