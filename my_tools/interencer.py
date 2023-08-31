from mmseg.apis import init_model, inference_model, show_result_pyplot
import os

config_path = '../configs/my_net/my_swin_vaihingen.py'
checkpoint_path = 'G:\\project\\mmsegmentation\\log\\vaihingen\\swin\\iter_80000.pth'
img_path = '../test_img/'

filedir = os.listdir(img_path)
# 从配置文件和权重文件构建模型
model = init_model(config_path, checkpoint_path, device='cuda:0')

for img in filedir:
    # 推理给定图像
    img_dir = os.path.join(img_path, img)
    result = inference_model(model, img_dir)
    # 展示分割结果
    #vis_image = show_result_pyplot(model, img_dir, result)

    # 保存可视化结果，输出图像将在 `workdirs/result.png` 路径下找到
    vis_iamge = show_result_pyplot(model, img_dir, result, out_file='test_img/result/'+ img,opacity =1)

    # 修改展示图像的时间，注意 0 是表示“无限”的特殊值
    #vis_image = show_result_pyplot(model, img_dir, result, wait_time=5)

