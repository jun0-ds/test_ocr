# -*- coding: utf-8 -*-
import sys
sys.path.append('./CLOVA_CRAFT')
sys.path.append('./CLOVA_OCR')
from time import time

from config import config_craft, config_ocr
from CLOVA_CRAFT import file_utils
from CLOVA_CRAFT import imgproc

from craft_one import CraftOne
from ocr_one import OcrOne

if __name__ == '__main__':

    # ================================================================================#
    # 0. define config for test
    # ================================================================================#
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    args_craft = Struct(**config_craft)
    opt = Struct(**config_ocr)

    # ================================================================================#
    # 1. load images from directory
    # ================================================================================#
    image_list, _, _ = file_utils.get_files(args_craft.test_folder)
    images = [imgproc.loadImage(image_path) for image_path in image_list]

    # ================================================================================#
    # 2. set class and initiate model
    # ================================================================================#
    CO = CraftOne()
    OO = OcrOne()
    CO = CraftOne(cuda=False)
    OO = OcrOne(cuda=False)
    for image, image_path in zip(images, image_list):
        # ================================================================================#
        # 3. get bbox points from CLOVA_CRAFT
        # ================================================================================#
        start = time()
        bboxes, polys, score_text = CO.main(image)
        print(f'craft one image {time() - start}')

        # ================================================================================#
        # 4. crop each bbox and extract text
        # ================================================================================#
        start = time()
        ocr_result = OO.main(image, image_path, bboxes)
        print(f'ocr one image {time() - start}')

        # ================================================================================#
        # 5. import result to somewhere
        # ================================================================================#
