"""
Chart image enhancer module
"""

import os
import random
import logging
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from pydantic import BaseModel
from .chart import Chart
from ..constants import IMAGE_SIZE, LOGO_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancerConfig(BaseModel):
    """
    config for image enhancer
    """
    use_clip: bool
    add_logo_prob: float
    add_title_prob: float
    clip_chart_prob: float
    hue_shift_prob: float
    contrast_prob: float
    brightness_prob: float
    saturation_prob: float

class ChartEnhancer:
    """
    enhancer for chart image
    """
    def __init__(self, config: EnhancerConfig):
        self.config = config

    def adjust_size(self, chart: Chart) -> Chart:
        """
        adjust size of the image to the setting size to fit the model input size
        """
        if self.config.use_clip:
            image_ratio = chart.image.width / chart.image.height
            target_ratio = IMAGE_SIZE[0] / IMAGE_SIZE[1]
            if image_ratio > target_ratio: # image is wider than target
                new_width = chart.image.height * target_ratio
                delta_width = chart.image.width - new_width
                chart.image = chart.image.crop((delta_width / 2, 0, new_width + delta_width / 2, chart.image.height))
            elif image_ratio < target_ratio: # image is taller than target
                new_height = chart.image.width / target_ratio
                delta_height = chart.image.height - new_height
                chart.image = chart.image.crop((0, delta_height / 2, chart.image.width, new_height + delta_height / 2))
            #else image keep the same ratio, do nothing

        chart.image = chart.image.resize((IMAGE_SIZE[0], IMAGE_SIZE[1]), Image.Resampling.LANCZOS)
        return chart

    def clip_chart_area(self, chart: Chart) -> Chart:
        """
        clip the chart area to simulate the clip operation for improving the model generalization
        """
        clip_ratio = random.uniform(0.6, 0.9)
        image_ratio = chart.image.width / chart.image.height
        if image_ratio > 3 or image_ratio < 0.3:
            return chart # do not clip the image if it is too wide or too narrow
        target_ratio = IMAGE_SIZE[0] / IMAGE_SIZE[1]
        if image_ratio < target_ratio: # image is taller than target
            new_width = int(chart.image.width * clip_ratio)
            new_height = int(new_width / target_ratio)
            max_start_width = chart.image.width - new_width - 1
            max_start_height = chart.image.height - new_height - 1
            start_width = random.randint(0, max_start_width)
            start_height = random.randint(0, max_start_height)
            chart.image = chart.image.crop((start_width, start_height, start_width + new_width, start_height + new_height))
            return chart

        # image is wider than target
        new_height = int(chart.image.height * clip_ratio)
        new_width = int(new_height / target_ratio)
        max_start_height = chart.image.height - new_height - 1
        max_start_width = chart.image.width - new_width - 1
        start_height = random.randint(0, max_start_height)
        start_width = random.randint(0, max_start_width)
        chart.image = chart.image.crop((start_width, start_height, start_width + new_width, start_height + new_height))
        return chart

    def change_chart_hue(self, chart: Chart) -> Chart:
        """
        change the hue of the chart for improving the model generalization
        """
        hue_shift = random.randint(-30, 30)
        hsv_image = chart.image.convert('HSV')
        pixels = list(hsv_image.getdata())

        new_pixels = []
        for h, s, v in pixels:
            new_h = (h + hue_shift) % 256
            new_pixels.append((new_h, s, v))

        new_hsv = Image.new('HSV', hsv_image.size)
        new_hsv.putdata(new_pixels)

        chart.image = new_hsv.convert('RGB')
        return chart

    def change_chart_contrast(self, chart: Chart) -> Chart:
        """
        change the contrast of the chart for improving the model generalization
        """
        contrast = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Contrast(chart.image)
        chart.image = enhancer.enhance(contrast)
        return chart

    def change_chart_brightness(self, chart: Chart) -> Chart:
        """
        change the brightness of the chart for improving the model generalization
        """
        brightness = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Brightness(chart.image)
        chart.image = enhancer.enhance(brightness)
        return chart

    def change_chart_saturation(self, chart: Chart) -> Chart:
        """
        change the saturation of the chart for improving the model generalization
        """
        saturation = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Color(chart.image)
        chart.image = enhancer.enhance(saturation)
        return chart

    def add_logo_watermark(self, chart: Chart) -> Chart:
        """
        add the logo watermark to the chart to simulate different organizations
        """
        logo_path = os.path.join(LOGO_DIR, random.choice(os.listdir(LOGO_DIR)))
        logo = Image.open(logo_path)
        
        # Convert logo to RGBA if it isn't already
        if logo.mode != 'RGBA':
            logo = logo.convert('RGBA')

        # Resize logo to 1/2 of the image size
        logo = logo.resize((chart.image.width // 2, chart.image.height // 2))

        corner_index = random.randint(0, 3) # stand for top-left, top-right, bottom-left, bottom-right
        if corner_index == 0:
            chart.image.paste(logo, (0, 0))
        elif corner_index == 1:
            chart.image.paste(logo, (chart.image.width - logo.width, 0))
        elif corner_index == 2:
            chart.image.paste(logo, (0, chart.image.height - logo.height))
        else:
            chart.image.paste(logo, (chart.image.width - logo.width, chart.image.height - logo.height))
        return chart

    def add_chart_title(self, chart: Chart) -> Chart:
        """
        add the chart title to the chart to simulate different charts
        """
        if chart.metadata.zh_name is None:
            return chart

        title_str = chart.construct_title()
        draw = ImageDraw.Draw(chart.image)

        # Try to use a larger font for title
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/unifont/unifont_sample.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()

        # Calculate text size and center position
        text_bbox = draw.textbbox((0, 0), title_str, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Center horizontally, place at top with margin
        x = (chart.image.width - text_width) // 2
        y = 10
        padding = 10  # Padding around text

        # Add semi-transparent background
        overlay = Image.new('RGBA', chart.image.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        # Draw background rectangle with semi-transparent white
        background_bbox = (
            x - padding,
            y - padding,
            x + text_width + padding,
            y + text_height + padding
        )
        overlay_draw.rectangle(background_bbox, fill=(255, 255, 255, 180))

        # Draw title text
        fill_color = random.choice([(0, 0, 0, 255), (255, 0, 0, 255)])  # 只使用黑色或红色文字
        overlay_draw.text((x, y), title_str, fill=fill_color, font=font)

        # Composite with original image
        if chart.image.mode != 'RGBA':
            chart.image = chart.image.convert('RGBA')
        chart.image = Image.alpha_composite(chart.image, overlay)
        return chart

    def enhance(self, chart: Chart) -> Chart:
        """
        enhance the chart image by applying the enhancer config
        """
        # Apply clipping if enabled and randomly triggered
        if self.config.use_clip and random.random() < self.config.clip_chart_prob:
            chart = self.clip_chart_area(chart)
            logger.debug("Applied clipping")

        # First adjust the size to ensure consistent dimensions
        chart = self.adjust_size(chart)

        # Add logo watermark
        if random.random() < self.config.add_logo_prob:
            chart = self.add_logo_watermark(chart)
            logger.debug("Added logo")

        # Convert to RGBA mode for operations that need alpha channel
        if chart.image.mode != 'RGBA':
            chart.image = chart.image.convert('RGBA')

        # Apply hue shift
        if random.random() < self.config.hue_shift_prob:
            chart = self.change_chart_hue(chart)
            logger.debug("Applied hue shift")

        # Apply contrast adjustment
        if random.random() < self.config.contrast_prob:
            chart = self.change_chart_contrast(chart)
            logger.debug("Applied contrast")

        # Apply brightness adjustment
        if random.random() < self.config.brightness_prob:
            chart = self.change_chart_brightness(chart)
            logger.debug("Applied brightness")

        # Apply saturation adjustment
        if random.random() < self.config.saturation_prob:
            chart = self.change_chart_saturation(chart)
            logger.debug("Applied saturation")

        # Add chart title
        if random.random() < self.config.add_title_prob:
            chart = self.add_chart_title(chart)
            logger.debug("Added title")

        # Ensure final image is in RGB mode
        if chart.image.mode != 'RGB':
            chart.image = chart.image.convert('RGB')

        return chart

    def __call__(self, chart: Chart) -> Chart:
        return self.enhance(chart)

    def __str__(self) -> str:
        return self.config.model_dump_json()

EnhancerConfigPresets = {
    "None": EnhancerConfig(
        use_clip=True,
        add_logo_prob=0.0,
        add_title_prob=0.0,
        clip_chart_prob=0.0,
        hue_shift_prob=0.0,
        contrast_prob=0.0,
        brightness_prob=0.0,
        saturation_prob=0.0
    ),
    "HighClipLowNoise": EnhancerConfig(
        use_clip=True,
        add_logo_prob=0.2,
        add_title_prob=0.2,
        clip_chart_prob=0.8,  # high clip probability
        hue_shift_prob=0.1,   # low image variation
        contrast_prob=0.1,
        brightness_prob=0.1,
        saturation_prob=0.1
    ),
    "MediumWatermarkLowClip": EnhancerConfig(
        use_clip=False,
        add_logo_prob=0.5,    # medium watermark probability
        add_title_prob=0.4,
        clip_chart_prob=0.2,  # low clip probability
        hue_shift_prob=0.2,
        contrast_prob=0.2,
        brightness_prob=0.2,
        saturation_prob=0.2
    ),
    "HighWatermarkStable": EnhancerConfig(
        use_clip=True,
        add_logo_prob=0.7,    # high watermark probability
        add_title_prob=0.8,
        clip_chart_prob=0.1,  # low clip probability
        hue_shift_prob=0.1,   # low image variation
        contrast_prob=0.1,
        brightness_prob=0.1,
        saturation_prob=0.1
    ),
    "BalancedEnhance": EnhancerConfig(
        use_clip=True,
        add_logo_prob=0.4,    # balanced watermark probability
        add_title_prob=0.4,
        clip_chart_prob=0.4,  # balanced clip probability
        hue_shift_prob=0.3,   # moderate image variation
        contrast_prob=0.3,
        brightness_prob=0.3,
        saturation_prob=0.3
    ),
    "HighColorVariation": EnhancerConfig(
        use_clip=True,
        add_logo_prob=0.2,
        add_title_prob=0.2,
        clip_chart_prob=0.2,
        hue_shift_prob=0.8,   # high hue variation
        contrast_prob=0.2,
        brightness_prob=0.4,
        saturation_prob=0.4
    ),
    "NightModeSimulation": EnhancerConfig(
        use_clip=True,
        add_logo_prob=0.3,
        add_title_prob=0.3,
        clip_chart_prob=0.2,
        hue_shift_prob=0.4,
        contrast_prob=0.8,    # high contrast
        brightness_prob=0.9,  # high brightness adjustment
        saturation_prob=0.5
    ),
    "PrintStyleSimulation": EnhancerConfig(
        use_clip=True,
        add_logo_prob=0.9,    # high watermark prob
        add_title_prob=0.9,
        clip_chart_prob=0.1,
        hue_shift_prob=0.2,
        contrast_prob=0.4,    # medium contrast
        brightness_prob=0.3,
        saturation_prob=0.2   # low saturation
    ),
    "WeatherAppStyle": EnhancerConfig(
        use_clip=True,
        add_logo_prob=0.6,    # high watermark probability
        add_title_prob=0.7,   # high title
        clip_chart_prob=0.5,  # medium clip
        hue_shift_prob=0.3,
        contrast_prob=0.5,    # medium contrast
        brightness_prob=0.4,
        saturation_prob=0.6   # high saturation
    ),
    "PresentationReady": EnhancerConfig(
        use_clip=True,
        add_logo_prob=0.95,   # almost always add watermark
        add_title_prob=0.95,  # almost always add title
        clip_chart_prob=0.3,  # low clip
        hue_shift_prob=0.2,   # low hue variation
        contrast_prob=0.4,    # medium contrast
        brightness_prob=0.3,  # low brightness adjustment
        saturation_prob=0.4   # medium saturation
    ),
    "ExtremeVariation": EnhancerConfig(
        use_clip=True,
        add_logo_prob=1.0,
        add_title_prob=1.0,
        clip_chart_prob=0.9,
        hue_shift_prob=0.9,
        contrast_prob=0.9,
        brightness_prob=0.9,
        saturation_prob=0.9
    )
}
