# Weather Chart Classification

天气图表参数类型分类系统，基于ECMWF图表样式进行天气图表的自动化筛选和爬取。

## 功能特性

### 🔄 双模式操作
- **本地模式**: 对本地HTML文件进行筛选操作
- **远程模式**: 直接从[ECMWF Charts网站](https://charts.ecmwf.int/)爬取数据

### 🎯 智能筛选
- 支持多种筛选类别：地面/大气、产品类型、参数等
- 批量筛选操作
- 筛选状态管理和清除功能

### 🕷️ 网站爬取
- 自动解析ECMWF Charts网站结构
- 提取图表元数据和图像URL
- 识别可用的筛选选项

### 🔧 统一架构
- 集中式WebDriver管理
- 模式间无缝切换
- 会话数据保存和恢复

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- `selenium>=4.0.0` - 浏览器自动化
- `webdriver-manager>=3.8.0` - WebDriver管理
- `requests>=2.28.0` - HTTP请求
- `beautifulsoup4>=4.11.0` - HTML解析

## 快速开始

### 基本使用

```python
from crawler.crawler import Crawler

# 创建爬虫实例
with Crawler(headless=False) as crawler:
    # 本地HTML文件筛选
    crawler.load_local_gallery("gallary/gallary.html")

    # 应用筛选条件
    filters = {
        'parameters': ['Wind', 'Temperature'],
        'surface_atmosphere': ['Surface']
    }
    results = crawler.apply_filters(filters)

    # 切换到远程模式
    crawler.switch_mode('remote')

    # 提取网站元数据
    metadata = crawler.extract_gallery_metadata()
```

### 运行完整示例

```bash
python example_usage.py
```

## 项目结构

```
weatherchart_classification/
├── crawler/
│   ├── crawler.py           # 主爬虫类
│   ├── selector.py          # 本地HTML筛选器
│   ├── gallary_crawler.py   # 远程网站爬虫
│   └── gallery_filter_demo.py  # 筛选演示
├── gallary/
│   └── gallary.html         # 本地HTML文件
├── example_usage.py         # 综合使用示例
├── test_selector.py         # 测试脚本
└── requirements.txt         # 依赖列表
```

## 核心组件

### 1. Crawler (主控制器)
- 管理WebDriver生命周期
- 协调各个组件
- 提供统一的操作接口

### 2. GallerySelector (本地筛选器)
- 解析本地HTML文件
- 执行checkbox点击操作
- 管理筛选状态

### 3. GallaryCrawler (远程爬虫)
- 连接ECMWF Charts网站
- 解析网站结构
- 提取图表和筛选信息

## 支持的筛选类别

### 地面/大气 (Surface/Atmosphere)
- Surface: 地面数据
- Atmosphere: 大气数据

### 产品类型 (Product Type)
- Control Forecast (ex-HRES): 控制预报
- Ensemble forecast (ENS): 集合预报
- AIFS Single/Ensemble: AI预报系统
- 实验性机器学习模型

### 参数 (Parameters)
- 风场 (Wind)
- 温度 (Temperature)
- 降水 (Precipitation)
- 海平面气压 (Mean sea level pressure)
- 位势高度 (Geopotential)
- 云量 (Cloud)
- 湿度 (Humidity)
- 水汽 (Water vapour)
- 雪 (Snow)
- 海浪 (Ocean waves)
- 地面特征 (Surface characteristics)

## 使用场景

### 天气分析场景

#### 降水分析
```python
precipitation_filters = {
    'parameters': ['Precipitation', 'Cloud', 'Humidity'],
    'surface_atmosphere': ['Atmosphere']
}
```

#### 地面天气条件
```python
surface_filters = {
    'parameters': ['Temperature', 'Mean sea level pressure', 'Wind'],
    'surface_atmosphere': ['Surface'],
    'product_type': ['Control Forecast (ex-HRES)']
}
```

#### 海洋条件
```python
ocean_filters = {
    'parameters': ['Ocean waves', 'Surface characteristics'],
    'product_type': ['Ensemble forecast (ENS)']
}
```

## 输出数据

系统会生成以下文件：
- `local_gallery_session.json` - 本地会话数据
- `remote_gallery_session.json` - 远程会话数据
- `ecmwf_gallery_metadata.json` - ECMWF网站元数据

## 注意事项

1. **浏览器要求**: 需要安装Chrome浏览器
2. **网络连接**: 远程模式需要稳定的网络连接
3. **认证**: ECMWF网站可能需要登录才能访问完整功能
4. **性能**: 建议使用headless模式提高性能
5. **资源管理**: 使用context manager确保资源正确清理

## 故障排除

### 常见问题

1. **WebDriver错误**
   - 确保Chrome浏览器已安装
   - 检查selenium版本兼容性

2. **网站连接失败**
   - 检查网络连接
   - 确认ECMWF网站可访问
   - 可能需要VPN或代理

3. **筛选器失效**
   - 检查HTML文件是否正确
   - 验证CSS选择器是否匹配

### 调试模式

设置 `headless=False` 可以观察浏览器操作过程：

```python
with Crawler(headless=False, wait_timeout=20) as crawler:
    # 调试操作
    pass
```

## 贡献指南

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件
