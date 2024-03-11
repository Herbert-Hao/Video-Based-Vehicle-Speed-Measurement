import argparse
import os
import cv2
import glob
import numpy as np
import torch
import math
from PIL import Image
from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder

import shutil
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized


DEVICE = 'cpu'
i = 0


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, information1, colors):
    global i
    i = i + 1
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    for a in information1:
        label = '%.2fkm/h' % a[1]
        plot_one_box(a[0][0], flo, label=label, color=colors[int(a[0][1])], line_thickness=3)
    img_flo = np.concatenate([img, flo], axis=0)

    b, g, r = cv2.split(img_flo)
    img_flo = cv2.merge([r, g, b])
    cv2.imwrite('output/{}.png'.format(i), img_flo)


def dandk(x, y, h=8.18, b=17, x0=955.25, y0=584.69, fx=2524.32, fy=2523.86):
    d = h / math.tan((math.pi*b/180) + math.atan((y-y0)/fy))
    k = math.sqrt(h*h + d*d) * (x-x0) * fy / (math.sqrt((y-y0)*(y-y0) + fy*fy) * fx)
    return d, k


def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.save_dir, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')
    # global centers
    # Initialize
    set_logging()
    device = select_device(opt.device)
    fps = int(input("请输入视频帧率："))
    if os.path.exists(out):  # output dir
        shutil.rmtree(out)  # delete dir
    os.makedirs(out)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    # t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    modelr = torch.nn.DataParallel(RAFT(opt))  # modelr是raft的预训练权重
    modelr.load_state_dict(torch.load(opt.modelr))
    modelr = modelr.module
    modelr.to(DEVICE)
    modelr.eval()
    with torch.no_grad():
        images = glob.glob(os.path.join(opt.path, '*.png')) + \
                 glob.glob(os.path.join(opt.path, '*.jpg'))

        images = sorted(images, key=lambda x: int(x.split("demo-frames\\")[1].split('.png')[0]))  # demo-frames文件夹下图片排序
        for (path, img, im0s, vid_cap), (imfile1, imfile2) in zip(dataset, zip(images[:-1], images[1:])):  # yolo和raft要检测的图片对应起来
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            centers = []
            speed = []
            information = []
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                s += '%gx%g ' % img.shape[2:]  # print string
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        xyxy = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                        c1, c2 = (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])  # xyxy是检测框左上角、右下角横纵坐标
                        center = int((c2[1] + c1[1]) / 2), int((c2[0] + c1[0]) / 2)  # center是（y, x)，因为图像坐标和张量索引不一样
                        # center = int((c2[1] + c1[1]) / 2), int((c2[0] + c1[0]) / 2)
                        width = abs(c1[0] - c2[0])
                        information.append([xyxy, cls])  # information列表，yolo坐标，分类
                        centers.append([center])  # centers列表，检测框中心坐标、检测框像素宽度、车辆真实宽度
                        # ld = (c1[0], c2[1])
                        # rd = (c2[0], c2[1])
                        # print("左下点的坐标为:" + str(ld) + "，右下点的坐标为" + str(rd) + "，中心点的坐标为" + str(center))

            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = modelr(image1, image2, iters=20, test_mode=True)  # torch.Size([1, 2, 135, 240]) torch.Size([1, 2, 1080, 1920])
            f1, f2 = torch.split(flow_up, 1, 1)  # torch.Size([1, 1, 1080, 1920])
            f1 = torch.squeeze(f1)  # torch.Size([1080, 1920])
            f2 = torch.squeeze(f2)
            # print(f1, f2)
            # f = torch.square(f1) + torch.square(f2)
            # f = torch.sqrt(torch.square(f1) + torch.square(f2))  # torch.Size([1080, 1920])
            # print(f.size())
            for work in centers:  # centers列表，检测框中心坐标、检测框像素宽度、车辆真实宽度
                # distance = f[work[0]].item()
                xr = f1[work[0]].item()
                yr = f2[work[0]].item()
                y1, x1 = map(float, work[0])
                x2, y2 = tuple(map(sum, zip((xr, yr), (x1, y1))))
                d1, k1 = dandk(x1, y1)
                d2, k2 = dandk(x2, y2)
                # print(d1,k1,d2,k2)
                l = math.sqrt((d2-d1)*(d2-d1)+(k2-k1)*(k2-k1))
                speed.append(float('%.2f' % (l * fps / 5 * 3.6)))
            information1 = list(zip(information, speed))  # information1列表， [[[662, 85, 720, 123], 2], 1.51],用来画框和label
            viz(image1, flow_up, information1, colors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelr', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='demo-frames', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--classes', nargs='+', type=int, default=(2, 5, 7), help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
    # demo(opt)
