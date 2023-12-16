import os
from PIL import Image as PILImage, PngImagePlugin
import piexif
import piexif.helper


class Image:

    def __init__(self, image: PILImage, prompt_text, filename_prefix, settings: dict):
        self.image = image
        self.prompt_text = prompt_text
        self.filename_prefix = filename_prefix
        self.settings = settings
        try:
            self.geninfo = f"{prompt_text}\nSteps: {settings['steps']}, Sampler: {settings['sampler']}, CFG scale: {settings['cfg_scale']}, Seed: {settings['seed']}, Size: {settings['width']}x{settings['height']}, Model: {settings['model_id']}"
        except KeyError as e:
            raise ValueError(f"Missing key in settings: {e}")

    def save(self, filename):
        """
        Saves image to filename, including geninfo as text information for generation info.
        For PNG images, geninfo is added to existing pnginfo dictionary using the pnginfo_section_name argument as key.
        For JPG images, there's no dictionary and geninfo just replaces the EXIF description.
        """

        extension = os.path.splitext(filename)[1]

        image_format = PILImage.registered_extensions()[extension]

        if extension.lower() == '.png':
            pnginfo_data = PngImagePlugin.PngInfo()
            for k, v in (self.geninfo or {}).items():
                pnginfo_data.add_text(k, str(v))
            self.image.save(filename, format=image_format, pnginfo=pnginfo_data)

        elif extension.lower() in (".jpg", ".jpeg", ".webp"):
            if self.image.mode == 'RGBA':
                self.image = self.image.convert("RGB")
            elif self.image.mode == 'I;16':
                self.image = self.image.point(lambda p: p * 0.0038910505836576).convert(
                    "RGB" if extension.lower() == ".webp" else "L")
            self.image.save(filename, format=image_format)
            if self.geninfo is not None:
                exif_bytes = piexif.dump({
                    "Exif": {
                        piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(self.geninfo or "", encoding="unicode")
                    },
                })
                piexif.insert(exif_bytes, filename)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")
