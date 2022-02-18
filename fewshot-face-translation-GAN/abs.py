import cv2
img1=cv2.imread("images/raw/1.jpg")
img2=cv2.imread("images/raw/1.jpg")
from utils.faceswap import FaceSwap
fs= FaceSwap()
abc=fs.transform(img1,img2)
print(abc)
