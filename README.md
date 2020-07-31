# Python snippets

### Download zipfile and decompress
```python
import os
import zipfile
from pathlib import Path

import requests

# URL, path(sample)
DATA_PATH = Path("data")
PATH = DATA_PATH / "PennFudanPed"
PATH.mkdir(parents=True, exist_ok=True)
URL = "https://www.cis.upenn.edu/~jshi/ped_html/"
FILENAME = "PennFudanPed.zip"
FOLDERNAME = os.path.splitext(FILENAME)[0]

# Download
if  (PATH / FILENAME).exists():
    print("Zipfile exists.")
else:
    print("Downloading now...")
    content = requests.get(URL + FILENAME).content
    with (PATH / FILENAME).open("wb") as f:
        f.write(content)
        print("The download is completed.")

# Decompress
if (PATH / FOLDERNAME).exists():
    print("Zipfile is already decompressed.")
else:
    print("Decompressing now...")
    with zipfile.ZipFile(PATH / FILENAME) as f:
        f.extractall(PATH)
        print("The decompression is completed.")
```


### 
```python

```

### 
```python

```

### 
```python

```