# Python snippets

# Programming competitions

### UnionFindTree ([参考](https://note.nkmk.me/python-union-find))
```python
class UnionFind():
    def __init__(self, n):
        self.n = n
        self.parents = [-1] * n

    def find(self, x):
        if self.parents[x] < 0:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return

        if self.parents[x] > self.parents[y]:
            x, y = y, x

        self.parents[x] += self.parents[y]
        self.parents[y] = x

    def size(self, x):
        return -self.parents[self.find(x)]

    def same(self, x, y):
        return self.find(x) == self.find(y)

    def members(self, x):
        root = self.find(x)
        return [i for i in range(self.n) if self.find(i) == root]

    def roots(self):
        return [i for i, x in enumerate(self.parents) if x < 0]

    def group_count(self):
        return len(self.roots())

    def all_group_members(self):
        return {r: self.members(r) for r in self.roots()}

    def __str__(self):
        return '\n'.join('{}: {}'.format(r, self.members(r)) for r in self.roots())
```

# Utils

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

### Generate gif from png
```python
from PIL import Image
import glob

files = sorted(glob.glob('./*.png'))
images = list(map(lambda file: Image.open(file), files))

images[0].save('out.gif', save_all=True, append_images=images[1:], duration=400, loop=0)
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