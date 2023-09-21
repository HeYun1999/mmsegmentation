import torch
import torch.nn as nn

# 示例数据，假设您有原始标签和模型输出的 logits
original_labels = torch.tensor([[[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]],

                                [[3, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]]]
                               )

logits = torch.randn((2, 4, 3, 3))  # 模型输出的 logits，假设有4个类别

# 预处理标签，将相邻像素值相同的位置设置为-1
processed_labels = original_labels.clone()  # 创建预处理后的标签副本
processed_labels = processed_labels.unsqueeze(1)
original_labels_clone = processed_labels
# 使用卷积操作检查相邻像素
conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
conv.weight.data = torch.tensor([[[[1, 1, 1], [1, -8, 1], [1, 1, 1]]]], dtype=torch.float32)

# 将标签转换为张量并添加通道维度
labels_tensor = processed_labels.float()

# 应用卷积操作
processed_labels = conv(labels_tensor)

# 将相同的位置设置为-1
original_labels_clone = torch.where(processed_labels == 0,-1,original_labels_clone)

original_labels = original_labels.squeeze(1)
# 定义损失函数，忽略标签为-1的像素
loss_function = nn.CrossEntropyLoss(ignore_index=-1)

# 计算损失
logits = logits.view(-1, 4, 3, 3)  # 调整 logits 形状以匹配标签

loss = loss_function(logits, processed_labels)

print("Loss for pixels with different neighboring labels:", loss.item())
