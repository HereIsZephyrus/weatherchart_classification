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
        target_ratio = IMAGE_SIZE[0] / IMAGE_SIZE[1]
        if image_ratio > target_ratio: # image is wider than target
            new_width = chart.image.height * clip_ratio
            new_height = chart.image.width / clip_ratio
            max_start_width = chart.image.width - new_width-1
            max_start_height = chart.image.height - new_height-1
            start_width = random.randint(0, max_start_width)
            start_height = random.randint(0, max_start_height)
            chart.image = chart.image.crop((start_width, start_height, start_width + new_width, start_height + new_height))
            return chart

        # image is taller than target
        new_height = chart.image.width * clip_ratio
        new_width = chart.image.height / clip_ratio
        max_start_height = chart.image.height - new_height-1
        max_start_width = chart.image.width - new_width-1
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
        title_str = chart.construct_title()
        draw = ImageDraw.Draw(chart.image)

        # Try to use a larger font for title
        try:
            font = ImageFont.truetype("arial.ttf", 10)
        except (OSError, IOError):
            font = ImageFont.load_default()

        # Calculate text size and center position
        text_bbox = draw.textbbox((0, 0), title_str, font=font)
        text_width = text_bbox[2] - text_bbox[0]

        # Center horizontally, place at top with margin
        x = (chart.image.width - text_width) // 2
        y = 10

        # Add semi-transparent background
        overlay = Image.new('RGBA', chart.image.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        # Draw title text
        overlay_draw.text((x, y), title_str, fill=(0, 0, 0, 255), font=font)

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

        chart = self.adjust_size(chart)

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

        # Add logo watermark
        if random.random() < self.config.add_logo_prob:
            chart = self.add_logo_watermark(chart)
            logger.debug("Added logo: %s")

        # Add chart title
        if random.random() < self.config.add_title_prob:
            chart = self.add_chart_title(chart)
            logger.debug("Added title: %s", chart.en_name)

        return chart

    def __call__(self, chart: Chart) -> Chart:
        return self.enhance(chart)

    def __str__(self) -> str:
        return self.config.model_dump_json()
