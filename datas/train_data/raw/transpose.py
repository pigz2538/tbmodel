import os
import numpy as np

# 遍历文件夹内所有子文件夹
root_folder = './'

for folder_name in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, folder_name)
    if os.path.isdir(folder_path):
        # 读取bands.npy文件并转置
        bands_file = os.path.join(folder_path, 'bands.npy')
        if os.path.exists(bands_file):
            bands_data = np.load(bands_file)
            transposed_bands_data = np.transpose(bands_data)

            # 重新写入到源文件中（覆盖）
            np.save(bands_file, transposed_bands_data)
            print(f'Transposed and saved bands.npy in {folder_path}')
