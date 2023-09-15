
#当第二个参数大于0或者小于0时，输出(800, 601, 3)
#当第二个参数为0是，输出(800, 601)，此时读出的是灰度图
from PIL import Image

image =Image.open(r'G:\project\mmsegmentation\data\Taiyuan_city\ann_dir\train\crop_700.png')
print(image)
print(image.size)
images = np.asarray(image)#转化成数组以后，iamges中存储的是图片的像素值。
print(images)
