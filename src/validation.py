import io

from PIL import Image, UnidentifiedImageError

MAX_BYTES = 10 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class ImageError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def validate_image(data: bytes) -> Image.Image:
    if not data:
        raise ImageError(400, "Empty file.")
    if len(data) > MAX_BYTES:
        raise ImageError(413, f"Image too large (max {MAX_BYTES // 1024 // 1024} MB).")
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data))
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise ImageError(415, "Not a readable image. Upload a JPG, PNG or WEBP photo.")
    if img.format not in ALLOWED_FORMATS:
        raise ImageError(415, f"Unsupported format {img.format}. Use JPG, PNG or WEBP.")
    if min(img.size) < 64:
        raise ImageError(400, "Image is too small to diagnose (min 64x64 pixels).")
    return img
