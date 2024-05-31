"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT

from collections import OrderedDict

from char_count_validator_modified import char_count_validator


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--char_bound_upper', default=7, type=int, help='char bound upper')
parser.add_argument('--char_bound_left', default=6, type=int, help='char bound left')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
parser.add_argument('--result_folder', default='result/', type=str, help='folder path to save result images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()


""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)

result_folder = args.result_folder
if not os.path.isdir(result_folder):
    os.makedirs(result_folder)

polys_folder = './poly_results'
if not os.path.isdir(polys_folder):
    os.makedirs(polys_folder)

def test_net(net, image, text_threshold, link_threshold, low_text, char_bound_upper, char_bound_left, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)
        
    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()
    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, box_labels, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    # box_labels cv2.connectedcomponentsden cixandir.
    char_boxes, char_labels = craft_utils.getCharBoxes_core(score_text, text_threshold, low_text, char_bound_upper, char_bound_left)
    
    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    char_boxes = craft_utils.adjustResultCoordinates(char_boxes, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    ret, text_score = cv2.threshold(score_text, low_text, 1, 0)
    ret, link_score = cv2.threshold(score_link, link_threshold, 1, 0)
    render_img2 = np.hstack((text_score, link_score))
    ret_score_link = imgproc.cvt2HeatmapImg(render_img2)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    ret_score_comb = imgproc.cvt2HeatmapImg(text_score_comb)

    if args.show_time : print("\n\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text, ret_score_link, ret_score_comb, char_boxes

def crop_polygon(image, polygon):
    """
    Crop a polygon (quadrilateral) from the given image and fill the background with white.
    """
    # Create a mask for the polygon
    mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(int)], 255)
    
    # Extract the bounding box of the polygon
    x, y, w, h = cv2.boundingRect(polygon)
    
    # Create a white background image
    background = np.ones_like(image, dtype=np.uint8) * 255
    
    # Copy the cropped region from the original image to the background
    background[y:y+h, x:x+w] = image[y:y+h, x:x+w]
    
    # Apply the mask to the background to fill the outside region with white
    result = cv2.bitwise_and(background, background, mask=mask)
    
    return result

def crop_polygon2(image, polygon, expansion_factor=1.2):
    """
    Crop a polygon (quadrilateral) from the given image and fill the background with white.
    """
    # Expand the bounding box around the polygon
    x, y, w, h = cv2.boundingRect(polygon)
    center_x = x + w / 2
    center_y = y + h / 2
    expanded_w = int(w * expansion_factor)
    expanded_h = int(h * expansion_factor)
    x = max(int(center_x - expanded_w / 2), 0)
    y = max(int(center_y - expanded_h / 2), 0)
    expanded_w = min(expanded_w, image.shape[1] - x)
    expanded_h = min(expanded_h, image.shape[0] - y)
    
    # Create a white background image
    background = np.ones((expanded_h, expanded_w, image.shape[2]), dtype=np.uint8) * 255
    
    # Copy the cropped region from the original image to the background
    roi = image[y:y+expanded_h, x:x+expanded_w]
    background[:roi.shape[0], :roi.shape[1]] = roi
    
    return background


if __name__ == '__main__':
    # load net
    net = CRAFT()     # initialize

    # print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    font_name = args.test_folder.split('/')[-1]
    # load data
    for k, image_path in enumerate(image_list):
        # print("\nTest image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)
        # boxes, polys, ret_score_text, ret_score_link, ret_score_comb, char_boxes, ret_box_labels, ret_char_labels
        bboxes, polys, score_text, score_link, score_comb, char_boxes = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.char_bound_upper, args.char_bound_left, args.cuda, args.poly, refine_net)
        
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        mask_file2 = result_folder + "/res_" + filename + '_mask2.jpg'
        # cv2.imwrite(mask_file, score_text)
        # cv2.imwrite(mask_file2, score_link)

        # no need for word masks.
        # mask_file3 = result_folder + "/res_" + filename + '_mask3.jpg'
        # cv2.imwrite(mask_file3, score_comb)

        file_utils.saveResult(image_path, image[:,:,::-1], char_boxes, img_dirname=result_folder, box_dirname=f'char_boxes/{font_name}')
        # Iterate over polygons and crop them
        
        # for i, poly in enumerate(polys):
        #     cropped_image = crop_polygon(image, poly)
        #     cv2.imwrite(f"{polys_folder}/{filename}_textbox_{i}.png", cropped_image)
 
        def save_char_images(image_path, char_boxes, save_dir):
            # Load the original image
            img = cv2.imread(image_path)
            img_name = os.path.basename(image_path)
        
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Iterate over the character boxes and save each as a separate image
            for i, box in enumerate(char_boxes):
                # The box points are expected to be in a clockwise order starting from top-left
                x_min = int(min(box[:, 0]))
                y_min = int(min(box[:, 1]))
                x_max = int(max(box[:, 0]))
                y_max = int(max(box[:, 1]))

                # Extract the character region from the image
                char_img = img[y_min:y_max, x_min:x_max]

                # Save the character image
                img_file_name, img_file_ending = os.path.splitext(img_name)
                save_path = os.path.join(save_dir, f"{img_file_name}_{i}{img_file_ending}")
                cv2.imwrite(save_path, char_img)
                
        # save_char_images(image_path, char_boxes, f'letters/{font_name}' )
        # print(f'letter image saved for {image_path}')
    
    # print("\nelapsed time : {}s".format(time.time() - t))

    char_count_validator(font_name)