import os
import rasterio

def filter_nodata(hr_input_path,lr_input_path):

    count = 0
 
    # 遍历输入目录中的所有文件
    for filename in os.listdir(hr_input_path):
        if filename.endswith(".tif"):  # 确保处理的是 TIFF 文件
            file_path = os.path.join(hr_input_path, filename)
            with rasterio.open(file_path) as src:
                nodata_value = src.nodata
                if nodata_value is not None:
                    data = src.read(1)
                    if not (data == nodata_value).any():  # 检查第一个波段是否包含 nodata
                        print(f"File {filename} contains nodata value {nodata_value}")
                        count += 1
                        lr_filename = filename.replace("hr", "lr")
                        lr_file_path = os.path.join(lr_input_path, lr_filename)
                        print(f"Removing {file_path} and {lr_file_path}")    
                        os.remove(file_path)
                        os.remove(lr_file_path)

    print(f"Found {count} files with nodata values")
# 定义路径
hr_input_path = 'dataset/HR'
lr_input_path = 'dataset/LR'
# Found 7357 files with nodata values
# 调用函数
filter_nodata(hr_input_path,lr_input_path)