from torchstat import stat

model = model.alexnet()
stat(model,(3,224,224))