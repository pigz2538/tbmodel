import os
import json

# 读取 fermi.txt 中的内容并处理
with open("fermi.txt", "r") as file:
    lines = file.readlines()
    for line in lines:
        folder_name, fermi_level = line.split()
        folder_name = folder_name.strip()
        fermi_level = float(fermi_level.strip())

        folder_path = os.path.join(os.getcwd(), folder_name)
        json_file_path = os.path.join(folder_name, f"{folder_name}.json")
        
        if not os.path.exists(folder_path):
            print(f"Folder '{folder_name}' does not exist. Skipping...")
            continue

        # 读取并更新 json 文件
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            data["fermi_level"] = fermi_level
        
        # 写入更新后的内容
        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file)
