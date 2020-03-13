# -*- coding: utf-8 -*-
import sys

sys.path.append('./CLOVA_CRAFT')
sys.path.append('./CLOVA_OCR')
from collections import OrderedDict, namedtuple
import string

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from PIL import Image
import cv2
import numpy as np

from config import config_ocr
from converter import CTCLabelConverter, AttnLabelConverter
from model import Model
from dataset import RawDataset, AlignCollate


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class OcrOne:
    # ================================================================================#
    # 0. set attributes and load model
    # ================================================================================#
    def __init__(self, cuda=True):
        self.cuda = cuda
        for k, v in config_ocr.items():
            setattr(self, k, v)

        """ vocab / character number configuration """
        if self.sensitive:
            self.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
        with open('./data/all_korean_char.txt', 'r', encoding='utf-8-sig') as kor:
            korean_char = kor.read()
        self.character += korean_char

        cudnn.benchmark = True
        cudnn.deterministic = True
        self.num_gpu = torch.cuda.device_count()

        print(self.__dict__)
        opt = Struct(**config_ocr)

        """ model configuration """
        if 'CTC' in self.Prediction:
            self.converter = CTCLabelConverter(self.character)
        else:
            self.converter = AttnLabelConverter(self.character)
        opt.num_class = len(self.converter.character)
        self.num_class = len(self.converter.character)

        if self.rgb:
            self.input_channel = 3

        self.model = Model(opt)
        print('model input parameters', self.__dict__.keys())

        self.model = torch.nn.DataParallel(self.model)
        if self.cuda and torch.cuda.is_available():
            self.model = self.model.cuda()

            # load model
            print('loading pretrained model from %s' % self.saved_model)
            self.model.load_state_dict(torch.load(self.saved_model))
        else:
            print('loading pretrained model from %s' % self.saved_model)
            self.model.load_state_dict(torch.load(self.saved_model, map_location='cpu'))

    # ================================================================================#
    # np.array by cv2 to image object by pillow
    # ================================================================================#
    def array2img(self, index, croped):
        try:
            if self.rgb:
                img = Image.fromarray(croped).convert('RGB')  # for color image
            else:
                img = Image.fromarray(croped).convert('L')

        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.rgb:
                img = Image.new('RGB', (self.imgW, self.imgH))
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))

        return img

    # ================================================================================#
    # crop one image from given bboxes
    # ================================================================================#
    def cropedDataset(self, image, image_path, bboxes):
        crops = []
        # Define points in input image bbox: top-left, top-right, bottom-right, bottom-left
        for i, bbox in enumerate(bboxes):
            # Define corresponding points in output image
            W = max(bbox[:, 0]) - min(bbox[:, 0])
            H = max(bbox[:, 1]) - min(bbox[:, 1])
            pts1 = np.float32([[0, 0], [W, 0], [W, H], [0, H]])

            # Get perspective transform and apply it
            croped = cv2.warpPerspective(image, cv2.getPerspectiveTransform(bbox, pts1), (W, H))

            crops.append((self.array2img(i, croped), f'{image_path}_{i}'))

        return crops
    # ================================================================================#
    # extract text from all bboxes in one image
    # ================================================================================#
    def extract_text(self, data_loader):
        # predict
        self.model.eval()
        result_str = ''
        for image_tensors, image_path_list in data_loader:
            batch_size = image_tensors.size(0)
            with torch.no_grad():
                image = image_tensors
                if self.cuda:
                    image = image.cuda()
                # For max length prediction
                length_for_pred = torch.cuda.IntTensor([self.batch_max_length] * batch_size)
                text_for_pred = torch.cuda.LongTensor(batch_size, self.batch_max_length + 1).fill_(0)

            if 'CTC' in self.Prediction:
                preds = self.model(image, text_for_pred).log_softmax(2)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.permute(1, 0, 2).max(2)
                preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
                preds_str = self.converter.decode(preds_index.data, preds_size.data)

            else:
                preds = self.model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, length_for_pred)

            # print('-' * 80)
            # print(f'image_path\t\t\t\tpredicted_labels')
            # print('-' * 80)

            for img_name, pred in zip(image_path_list, preds_str):
                if 'Attn' in self.Prediction:
                    pred = pred[:pred.find('[s]')]  # prune after "end of sentence" token ([s])
                # print(f'{img_name}\t{pred}')
                result_str += f' {pred}'
        return result_str

    # ================================================================================#
    # extract text from all bboxes in one image
    # ================================================================================#
    def main(self, image, image_path, bboxes):
        # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
        AlignCollate_demo = AlignCollate(imgH=self.imgH, imgW=self.imgW, keep_ratio_with_pad=self.PAD)
        dataset = self.cropedDataset(image=image, image_path=image_path, bboxes=bboxes)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=len(dataset),
            shuffle=False,
            num_workers=int(self.workers),
            collate_fn=AlignCollate_demo, pin_memory=True)

        result_str = self.extract_text(data_loader)

        return result_str