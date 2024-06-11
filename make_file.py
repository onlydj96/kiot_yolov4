from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split


# 이미지 파일 경로
img_list = glob('build/darknet/x64/data/obj/*.jpg')

print(len(img_list))

train_img_list, test_img_list = train_test_split(img_list, test_size=0.1, random_state=42) # random_state는 임의로 선택

print(len(train_img_list), len(test_img_list))


with open('build/darknet/x64/data/train.txt', 'w') as f:
    f.write('\n'.join(train_img_list) + '\n')

with open('build/darknet/x64/data/test.txt', 'w') as f:
    f.write('\n'.join(test_img_list) + '\n')