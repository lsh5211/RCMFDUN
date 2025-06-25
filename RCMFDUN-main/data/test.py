import os, glob, random, torch, cv2
import numpy as np
from argparse import ArgumentParser
from RCMFDUN import Net
from utils import *
from skimage.metrics import structural_similarity as ssim
from time import time
from  thop import  profile
parser = ArgumentParser(description='RCMFDUN')
parser.add_argument('--epoch', type=int, default=600)
parser.add_argument('--phase_num', type=int, default=10)
parser.add_argument('--block_size', type=int, default=32)
parser.add_argument('--model_dir', type=str, default='model')
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--testset_name', type=str, default='Set11')
parser.add_argument('--result_dir', type=str, default='test_out')
parser.add_argument('--gpu_list', type=str, default='0')
parser.add_argument('--channel_number', type=int, default=32)
parser.add_argument('--cs_ratio', type=float, default=0.25)

args = parser.parse_args()
epoch = args.epoch
N_p = args.phase_num
B = args.block_size
channel_number = args.channel_number
cs_ratio = args.cs_ratio

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# fixed seed for reproduction
seed = 2023
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

N = B * B

model = Net(sensing_rate=cs_ratio, LayerNo=N_p, channel_number=channel_number)
model = torch.nn.DataParallel(model).to(device)

num_count = 0
num_params = 0
for para in model.parameters():
    num_count += 1
    num_params += para.numel()
    print('Layer %d' % num_count)
    print(para.size())
print("total para num: %d" % num_params)
print("layernum: %d"%num_count)
print("")

model_dir = './%s/mark10_32_ratio_%.2f_layer_%d_block_%d' % (args.model_dir, cs_ratio, N_p, B)

model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, epoch)))

# test set info
test_image_paths = glob.glob(os.path.join(args.data_dir, args.testset_name) + '/*')
test_image_num = len(test_image_paths)
Time_All = np.zeros([1, test_image_num], dtype=np.float32)

result_dir = os.path.join(args.result_dir, args.testset_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

def test():
    with torch.no_grad():
        PSNR_list, SSIM_list = [], []
        for i in range(test_image_num):
            image_path = test_image_paths[i]
            test_image = cv2.imread(image_path, 1)  # read test data from image file
            test_image_ycrcb = cv2.cvtColor(test_image, cv2.COLOR_BGR2YCrCb)

            img, old_h, old_w, img_pad, new_h, new_w = my_zero_pad(test_image_ycrcb[:, :, 0])
            img_pad = img_pad.reshape(1, 1, new_h, new_w) / 255.0  # normalization

            x_input = torch.from_numpy(img_pad)
            x_input = x_input.type(torch.FloatTensor).to(device)

            start = time()
            x_output = model(x_input)
            end = time()

            x_output = x_output.cpu().data.numpy().squeeze()
            x_output = np.clip(x_output[:old_h, :old_w], 0, 1).astype(np.float64) * 255.0

            PSNR = psnr(x_output, img)
            SSIM = ssim(x_output, img, data_range=255)

            print('[%d/%d] %s, PSNR: %.3f, SSIM: %.4f, Time:%.4f' % (
            i, test_image_num, image_path, PSNR, SSIM, (end - start)))

            test_name_split = os.path.split(image_path)
            test_image_ycrcb[:, :, 0] = x_output
            im_rec_rgb = cv2.cvtColor(test_image_ycrcb, cv2.COLOR_YCrCb2BGR)
            im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)
            resultName = "./%s/%s" % (result_dir, test_name_split[1])
            cv2.imwrite("%s_CSratio_%.2f_channel_number%d_epoch_%d_PSNR_%.3f_SSIM_%.4f.png" % (
                resultName,args.cs_ratio,channel_number,args.epoch, PSNR, SSIM), im_rec_rgb)

            PSNR_list.append(PSNR)
            SSIM_list.append(SSIM)
            Time_All[0, i] = end - start
    return float(np.mean(PSNR_list)), float(np.mean(SSIM_list))


avg_psnr, avg_ssim = test()
print('CS ratio is %.2f, avg PSNR is %.4f, avg SSIM is %.5f. avg Time: %.4f' % (
cs_ratio, avg_psnr, avg_ssim, np.mean(Time_All)))




#
# # 中间重建结果
# def test():
#     with torch.no_grad():
#         PSNR_list, SSIM_list = [], []
#         iter5_psnr_list, iter5_ssim_list = [], []  # 第5层统计
#
#         for i in range(test_image_num):
#             image_path = test_image_paths[i]
#             test_image = cv2.imread(image_path, 1)
#             test_image_ycrcb = cv2.cvtColor(test_image, cv2.COLOR_BGR2YCrCb)
#
#             img, old_h, old_w, img_pad, new_h, new_w = my_zero_pad(test_image_ycrcb[:, :, 0])
#             img_pad = img_pad.reshape(1, 1, new_h, new_w) / 255.0
#
#             x_input = torch.from_numpy(img_pad).type(torch.FloatTensor).to(device)
#
#             start = time()
#             x_output, feat_iter5 = model(x_input)
#             end = time()
#
#             # 最终输出
#             x_output = x_output.cpu().data.numpy().squeeze()
#             x_output = np.clip(x_output[:old_h, :old_w], 0, 1) * 255.0
#
#             # 第5次迭代输出
#             x_iter5 = feat_iter5.cpu().data.numpy().squeeze()
#             x_iter5 = np.clip(x_iter5[:old_h, :old_w], 0, 1) * 255.0
#
#             # PSNR/SSIM
#             PSNR = psnr(x_output, img)
#             SSIM = ssim(x_output, img, data_range=255)
#             PSNR_iter5 = psnr(x_iter5, img)
#             SSIM_iter5 = ssim(x_iter5, img, data_range=255)
#
#             print('[%d/%d] %s, PSNR: %.3f, SSIM: %.4f, Time:%.4f' %
#                   (i, test_image_num, image_path, PSNR, SSIM, (end - start)))
#             print(f'→ Iter5 PSNR: {PSNR_iter5:.3f}, SSIM: {SSIM_iter5:.4f}')
#
#             # 保存评估结果到 txt
#             metrics_path = os.path.join(result_dir, 'iter5_metrics.txt')
#             with open(metrics_path, 'a') as f:
#                 f.write('[%d/%d] %s\n' % (i, test_image_num, image_path))
#                 f.write('Final   PSNR: %.3f, SSIM: %.4f\n' % (PSNR, SSIM))
#                 f.write('Iter5   PSNR: %.3f, SSIM: %.4f\n\n' % (PSNR_iter5, SSIM_iter5))
#
#             # 保存最终输出图像
#             test_name_split = os.path.split(image_path)
#             test_image_ycrcb[:, :, 0] = x_output
#             im_rec_rgb = cv2.cvtColor(test_image_ycrcb, cv2.COLOR_YCrCb2BGR)
#             im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)
#             resultName = "./%s/%s" % (result_dir, test_name_split[1])
#             cv2.imwrite("%s_CSratio_%.2f_channel_number%d_epoch_%d_PSNR_%.3f_SSIM_%.4f.png" %
#                         (resultName, args.cs_ratio, channel_number, args.epoch, PSNR, SSIM), im_rec_rgb)
#
#             # 保存第5次迭代输出图像
#             iter5_save_path = "%s_iter5_epoch_%d.png" % (resultName, args.epoch)
#             x_iter5 = np.round(x_iter5).astype(np.uint8)
#             cv2.imwrite(iter5_save_path, x_iter5)
#
#             # 记录统计值
#             PSNR_list.append(PSNR)
#             SSIM_list.append(SSIM)
#             iter5_psnr_list.append(PSNR_iter5)
#             iter5_ssim_list.append(SSIM_iter5)
#             Time_All[0, i] = end - start
#
#         # 统计平均值
#         avg_psnr = float(np.mean(PSNR_list))
#         avg_ssim = float(np.mean(SSIM_list))
#         avg_time = float(np.mean(Time_All))
#         avg_iter5_psnr = float(np.mean(iter5_psnr_list))
#         avg_iter5_ssim = float(np.mean(iter5_ssim_list))
#
#         print('\n=== Summary ===')
#         print('CS ratio: %.2f' % cs_ratio)
#         print('Final   → avg PSNR: %.4f, avg SSIM: %.5f, avg Time: %.4f' %
#               (avg_psnr, avg_ssim, avg_time))
#         print('Iter5   → avg PSNR: %.4f, avg SSIM: %.5f' %
#               (avg_iter5_psnr, avg_iter5_ssim))
#
#         return avg_psnr, avg_ssim
#
#

#
# avg_psnr, avg_ssim = test()
# print('CS ratio is %.2f, avg PSNR is %.4f, avg SSIM is %.5f. avg Time: %.4f' % (
# cs_ratio, avg_psnr, avg_ssim, np.mean(Time_All)))