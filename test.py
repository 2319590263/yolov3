import cv2
import numpy as np
import torch
from torch.autograd import Variable
from net.yolo3 import yolov3
from config import Config
from utils import utils
import cv2
from shutil import copyfile

def get_test_input():
    img = cv2.imread("dog.jpg")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

if __name__ == "__main__":
    img = get_test_input()
    anchors = torch.tensor(Config["yolo"]["anchors"])
    yolo = yolov3(anchors,80)
    yolo.load_state_dict(torch.load(r"weight/yolo_weights.pth"))
    predict = yolo(img)
    predict = utils.non_max_suppression(predict, 80)
    copyfile("dog.jpg", "dog_predict.jpg")
    for i in predict:
        for j in i:
            image = cv2.imread(r'dog_predict.jpg')
            tangle = cv2.rectangle(image, (int(768 * j[0]), int(576 * j[1])), (int(768 * j[2]), int(576 * j[3])), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(tangle, f"{int(j[6])}:{round(float(j[4] * j[5]), 2)}", (int(768 * j[0]), int(576 * j[1])), font, 1, (0, 0, 0), 2, 1)
            cv2.imwrite(r'dog_predict.jpg', image)

    
