"""
Chart image enhancer module
"""

import os
import random
import logging
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from pydantic import BaseModel
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
        self.current_image = None

    def adjust_size(self, image: Image.Image) -> Image.Image:
        """
        adjust size of the image to the setting size to fit the model input size
        """
        if self.config.use_clip:
            image_ratio = image.width / image.height
            target_ratio = IMAGE_SIZE[0] / IMAGE_SIZE[1]
            if image_ratio > target_ratio: # image is wider than target
                new_width = image.height * target_ratio
                delta_width = image.width - new_width
                image = image.crop((delta_width / 2, 0, new_width + delta_width / 2, image.height))
            elif image_ratio < target_ratio: # image is taller than target
                new_height = image.width / target_ratio
                delta_height = image.height - new_height
                image = image.crop((0, delta_height / 2, image.width, new_height + delta_height / 2))
            #else image keep the same ratio, do nothing

        return image.resize((IMAGE_SIZE[0], IMAGE_SIZE[1]), Image.Resampling.LANCZOS)

    def clip_chart_area(self, image: Image.Image) -> Image.Image:
        """
        clip the chart area to simulate the clip operation for improving the model generalization
        """
        clip_ratio = random.uniform(0.6, 0.9)
        image_ratio = image.width / image.height
        target_ratio = IMAGE_SIZE[0] / IMAGE_SIZE[1]
        if image_ratio > target_ratio: # image is wider than target
            new_width = image.height * clip_ratio
            new_height = image.width / clip_ratio
            max_start_width = image.width - new_width-1
            max_start_height = image.height - new_height-1
            start_width = random.randint(0, max_start_width)
            start_height = random.randint(0, max_start_height)
            return image.crop((start_width, start_height, start_width + new_width, start_height + new_height))

        # image is taller than target
        new_height = image.width * clip_ratio
        new_width = image.height / clip_ratio
        max_start_height = image.height - new_height-1
        max_start_width = image.width - new_width-1
        start_height = random.randint(0, max_start_height)
        start_width = random.randint(0, max_start_width)
        return image.crop((start_width, start_height, start_width + new_width, start_height + new_height))

    def change_chart_hue(self, image: Image.Image) -> Image.Image:
        """
        change the hue of the chart for improving the model generalization
        """
        hue_shift = random.randint(-30, 30)
        hsv_image = image.convert('HSV')
        pixels = list(hsv_image.getdata())

        new_pixels = []
        for h, s, v in pixels:
            new_h = (h + hue_shift) % 256
            new_pixels.append((new_h, s, v))

        new_hsv = Image.new('HSV', hsv_image.size)
        new_hsv.putdata(new_pixels)

        return new_hsv.convert('RGB')

    def change_chart_contrast(self, image: Image.Image) -> Image.Image:
        """
        change the contrast of the chart for improving the model generalization
        """
        contrast = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(contrast)

    def change_chart_brightness(self, image: Image.Image) -> Image.Image:
        """
        change the brightness of the chart for improving the model generalization
        """
        brightness = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(brightness)

    def change_chart_saturation(self, image: Image.Image) -> Image.Image:
        """
        change the saturation of the chart for improving the model generalization
        """
        saturation = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(saturation)

    def add_logo_watermark(self, image: Image.Image) -> Image.Image:
        """
        add the logo watermark to the chart to simulate different organizations
        """
        logo_path = os.path.join(LOGO_DIR, random.choice(os.listdir(LOGO_DIR)))
        logo = Image.open(logo_path)
        corner_index = random.randint(0, 3) # stand for top-left, top-right, bottom-left, bottom-right
        if corner_index == 0:
            image.paste(logo, (0, 0))
        elif corner_index == 1:
            image.paste(logo, (image.width - logo.width, 0))
        elif corner_index == 2:
            image.paste(logo, (0, image.height - logo.height))
        else:
            image.paste(logo, (image.width - logo.width, image.height - logo.height))
        return image

    def add_chart_title(self, image: Image.Image, title: str) -> Image.Image:
        """
        add the chart title to the chart to simulate different charts
        """
        draw = ImageDraw.Draw(image)

        # Try to use a larger font for title
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()

        # Calculate text size and center position
        text_bbox = draw.textbbox((0, 0), title, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Center horizontally, place at top with margin
        x = (image.width - text_width) // 2
        y = 10

        # Add semi-transparent background
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        # Draw background rectangle
        overlay_draw.rectangle([x-5, y-2, x+text_width+5, y+text_height+2], 
                             fill=(255, 255, 255, 180))

        # Draw title text
        overlay_draw.text((x, y), title, fill=(0, 0, 0, 255), font=font)

        # Composite with original image
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        result = Image.alpha_composite(image, overlay)

        return result.convert('RGB')

    def enhance(self, image: Image.Image) -> Image.Image:
        """
        enhance the chart image by applying the enhancer config
        """
        enhanced_image = image.copy()

        # Apply clipping if enabled and randomly triggered
        if self.config.use_clip and random.random() < self.config.clip_chart_prob:
            enhanced_image = self.clip_chart_area(enhanced_image)
            logger.debug("Applied clipping")

        enhanced_image = self.adjust_size(enhanced_image)

        # Apply hue shift
        if random.random() < self.config.hue_shift_prob:
            enhanced_image = self.change_chart_hue(enhanced_image)
            logger.debug("Applied hue shift")

        # Apply contrast adjustment
        if random.random() < self.config.contrast_prob:
            enhanced_image = self.change_chart_contrast(enhanced_image)
            logger.debug("Applied contrast")

        # Apply brightness adjustment
        if random.random() < self.config.brightness_prob:
            enhanced_image = self.change_chart_brightness(enhanced_image)
            logger.debug("Applied brightness")

        # Apply saturation adjustment
        if random.random() < self.config.saturation_prob:
            enhanced_image = self.change_chart_saturation(enhanced_image)
            logger.debug("Applied saturation")

        # Add logo watermark
        if random.random() < self.config.add_logo_prob:
            enhanced_image = self.add_logo_watermark(enhanced_image)
            logger.debug("Added logo: %s")

        # Add chart title
        if random.random() < self.config.add_title_prob:
            titles = [
                "Weather Analysis", "Precipitation Forecast", "Temperature Map",
                "Wind Pattern", "Pressure Chart", "Satellite Image",
                "Radar Data", "Storm Track", "Climate Chart"
            ]
            title = random.choice(titles)
            enhanced_image = self.add_chart_title(enhanced_image, title)
            logger.debug("Added title: %s", title)

        return enhanced_image

    def __call__(self, image: Image.Image) -> Image.Image:
        return self.enhance(image)

    def __str__(self) -> str:
        return self.config.model_dump_json()
