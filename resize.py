from PIL import Image
import os, sys
from tqdm import tqdm

path = sys.argv[1]
dirs = os.listdir( path )
sz = (2000,2000)

def resize():
    for item in tqdm(dirs):
        if os.path.isfile(path+item):
            if item.endswith(".jpg"):
                print(item)
                im = Image.open(path+item)
                f, _ = os.path.splitext(path+item)
                im.thumbnail(sz, Image.Resampling.LANCZOS)
                im.save(f + '.jpg', quality=100)

resize()
