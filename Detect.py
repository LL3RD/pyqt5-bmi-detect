from Image_Processor import *
from Visual import *
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image/2.jpg')
P = Image_Processor()
k,m = P._detected(img)

shape2d = (m.shape[0], m.shape[1])
img_f = np.zeros(shape2d + (3,), dtype="float32")
img_f[:, :, :3] = [255, 255, 255]
v = Visual(img_f)
v.draw_keypoints_predictions(k)
v.draw_masks_predictions(m)
plt.axis('off')
plt.imshow(v.image)
plt.show()
