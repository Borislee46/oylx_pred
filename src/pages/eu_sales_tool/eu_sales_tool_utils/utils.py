import base64
from pathlib import Path


def get_image_as_base64(path):
    if not Path(path).exists():
        return None
    try:
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return None
