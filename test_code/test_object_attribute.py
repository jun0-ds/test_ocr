# ================================================================================#
# Object type attribute
# ================================================================================#
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


s = Struct(**{'a': 1, 'b': 2})

# ================================================================================#
# image crop test
# ================================================================================#
import os
import sys
sys.path.append('./CLOVA_CRAFT')
sys.path.append('./CLOVA_OCR')

import cv2
import numpy as np

from craft_one import CraftOne
# Set width and height of output image
W, H = 600, 200

# Load input image
image_path = './data/test_folder/img_4020.jpg'
img = cv2.imread(image_path)

# Define points in input image: top-left, top-right, bottom-right, bottom-left
CO = CraftOne()
bboxes, polys, score_text = CO.main(img)
for i, bbox in enumerate(bboxes):
    pts0 = bbox

    # Define corresponding points in output image
    W = max(bbox[:,0]) - min(bbox[:,0])
    H = max(bbox[:,1]) - min(bbox[:,1])
    pts1 = np.float32([[0, 0], [W, 0], [W, H], [0, H]])

    # Get perspective transform and apply it
    M = cv2.getPerspectiveTransform(pts0, pts1)
    result = cv2.warpPerspective(img, M, (W, H))

    # Save reult
    cv2.imwrite(f'result{i}.png', result)


class inside:
    def __init__(self):
        self.a = 1
        self.b = 1
        self.c = 1

class outside:
    def __init__(self):
        self.q = 2
        self.w = 2
        self.e = 2
    def test(self):
        In = inside()
        print(In.__dict__)
        print(self.__dict__)
        In.__dict__.update(self.__dict__)
        print(In.__dict__)

Out = outside()
Out.test()
Out.c