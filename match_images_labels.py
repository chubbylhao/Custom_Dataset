import os
import xml.etree.ElementTree as ET
import csv

# 文件组织方式如下
# datasets
#   |- images
#   |    |- 1.jpg
#   |    |- 2.jpg
#   |    |- ...
#   |- labels
#   |    |- 1.xml
#   |    |- 2.xml
#   |    |- ...
# match_images_labels.py
# images_labels.csv

image_path = './datasets/images'
label_path = './datasets/labels'
# list 长度为450（3类，一类150）
image_list = os.listdir(image_path)
label_list = os.listdir(label_path)

# 解析xml，拿到标签信息（对于GAN的训练，一张图片就只有一类，但一张图片有多个同类目标无所谓）
label_name = []
for label in label_list:
    tree = ET.parse(os.path.join(label_path, label))
    root = tree.getroot()
    name = root.find('object').find('name').text
    label_name.append(name)

with open('images_labels.csv', 'w', encoding='UTF-8', newline='') as f:
    writer = csv.writer(f)
    for match in zip(image_list, label_name):
        writer.writerow(match)

if __name__ == '__main__':
    pass
