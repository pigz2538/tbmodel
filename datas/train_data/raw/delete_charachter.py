import os

def remove_dash_from_cif_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.cif'):
                new_file_name = file.replace('-', '')  # 删除文件名中的 '-' 字符
                os.rename(os.path.join(root, file), os.path.join(root, new_file_name))

# 指定要遍历的目录
directory = './'

remove_dash_from_cif_files(directory)
