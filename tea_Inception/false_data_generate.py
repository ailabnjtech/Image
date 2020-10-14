import os
import random
import shutil
tarDir = "/home/njtech/TEA/FALSE"
count = {}
labels = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30,
          31, 34, 35, 36, 38, 39, 40, 42, 43, 51, 52, 59, 60, 61, 62, 63, 64]
for i in labels:
    for root, dirs, files in os.walk("/home/njtech/TEA/iphone7_cut"):
        label = root.split('/')[-1]
        if label != i & len(files) > 20:       # all images in a file
            tmp = random.sample(files, 20)
            # count += 1
            for file in tmp:
                srcDir = os.path.join(root, file)
                tarDir_new = os.path.join(tarDir, "FALSE" + i)
                os.makedirs(tarDir_new, exist_ok=True)
                shutil.copy(srcDir, tarDir_new)
