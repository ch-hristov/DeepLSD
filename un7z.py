import py7zr
with py7zr.SevenZipFile('train.7z', mode='r') as z:
    z.extractall('./data/engisense-lines/homographies')