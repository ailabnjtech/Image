import os, shutil
all_video_paths = []
count = 0
tarDir = "D:/my_python/TEA/dataset_all/"
for root, dirs, files in os.walk("D:\my_python\TEA\dataset"):
    if files:       # all images in a file
        count += 1
        label = root.split('\\')[-1]        # expression label
        # print(root)
        # print(label)
        for file in files:
            shutil.copy(os.path.join(root, file), tarDir + label + "/" + file)
        # all_video_paths.append(os.path.join(root, files[-1]))
print(count)