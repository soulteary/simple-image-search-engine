import os

image_directory = "images"

# 使用列表推导式获取目录中所有的 PNG 图片名称
png_files = [filename for filename in os.listdir(image_directory) if filename.endswith(".png")]

# 根据文件名中的数字部分进行排序
sorted_png_files = sorted(png_files, key=lambda x: int(x.split('-')[-1].split('.')[0]))

# 打印排序后的 PNG 图片名称列表
for idx, png_file in enumerate(sorted_png_files, start=1):
    print(f"{idx}: {png_file}")
