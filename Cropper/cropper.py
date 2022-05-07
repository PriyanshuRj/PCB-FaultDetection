from cropper_utils import main
import os
def starter(img):
    img_path = "imgs"
    path = os.path.join(img_path,img)
    filename = main(path=img_path,imgp=img)

