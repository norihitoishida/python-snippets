# Python snippets

### Download zipfile and decompress
```python
import os
import zipfile
from pathlib import Path

import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "PennFudanPed"
PATH.mkdir(parents=True, exist_ok=True)
URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
FILENAME = "PennFudanPed.zip"
FOLDERNAME = os.path.splitext(FILENAME)[0]

# Download
if  (PATH / FILENAME).exists():
    print("Zipfile exists.")
else:
    print("Downloading now...")
    content = requests.get(URL).content
    with (PATH / FILENAME).open("wb") as f:
        f.write(content)
        print("The download is completed.")

# Decompress
if (a).exists():
    print("Zipfile is already decompressed.")
else:
    print("Decompressing now...")
    with zipfile.ZipFile(PATH / FILENAME) as f:
        f.extractall(PATH)
        print("The decompression is completed.")
```