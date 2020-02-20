import os
import torch

import models

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

num_class = 5
ignore_label = 255

model = models.ERFNet(num_class)
input_mean = model.input_mean
input_std = model.input_std
policies = model.get_optim_policies()
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count())).cuda()

print('Loaded model!')

checkpoint = torch.load('trained/ERFNet_trained.tar')
start_epoch = checkpoint['epoch']
best_mIoU = checkpoint['best_mIoU']
torch.nn.Module.load_state_dict(model, checkpoint['state_dict'])

print('Loaded checkpoint!')

img_path = '/home/whizz/Desktop/ERFNet/list/driver_37_30frame/05181432_0203.MP4/00000.jpg'
img = cv2.imread(img_path).astype(np.float32)
image = image[240:, :, :]

model.eval()
input_var = torch.autograd.Variable(image, volatile=True)
output, output_exist = model(input_var)
output = F.softmax(output, dim=1)
pred = output.data.cpu().numpy()

