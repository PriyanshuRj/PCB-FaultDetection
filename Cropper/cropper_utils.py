import cv2
import os
import argparse
import numpy as np
import time
def main(img_path,imgp,h = 500):
    img = cv2.imread(img_path)
    shape = img.shape
    w = int((shape[1]*500)/shape[0])
    img = cv2.resize(img,(w,h))
    
    edged,img_singlet,warped = find_contors(img,h,w)
    save(warped,imgp)
    # while (True):
    #     cv2.imshow("image",img)
    #     cv2.imshow("canny",edged)
    #     cv2.imshow("Singlet",img_singlet)
    #     cv2.imshow("Wraped",warped)
    #     if cv2.waitKey(20) &0xFF ==27:
    #         break
    

def find_contors(img,h,w):
    image_size = [h,w]
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
    edged = cv2.Canny(blurred, 20, 100)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contor,left,right,top, bottom = max_area_contour(contours,image_size)
    img_singlet = img.copy()
    cv2.drawContours(img, contours, -1, (0,255,0), 3)
    # print("contor : ",contor)
    # print(contours[contor])
    cv2.drawContours(img_singlet, contours, contor, (0,255,0), 3)
    warped = rectanglePloter(left,right,top, bottom,img_singlet)

    return edged,img_singlet,warped

def max_area_contour(contours,image_size):
    max_area = 0
    l = r = t = b = 0
    for i in range(len(contours)):
        left = image_size[1]
        top = image_size[0]
        bottom = right = 0
        for j in contours[i]:
            if(j[0][0] <left):
                left = j[0][0]
            if(j[0][0] >right):
                right = j[0][0]
            if(j[0][1] <top):
                top = j[0][1]
            if(j[0][1] >bottom):
                bottom = j[0][1]
        area_contor = (bottom-top)*(right-left)
        if(area_contor>max_area):
            max_area = area_contor
            l = left
            r = right
            t = top
            b = bottom
            cont = i
            
    return cont,l,r,t, b

            
def rectanglePloter(left,right,top, bottom,img):
    cv2.rectangle(img,(left,top),(right,bottom),(255,250,180),4)
    warped = prespectiverestore(img,[[left,top],[right,top],[right,bottom],[left,bottom]])
    return warped
     
    
def prespectiverestore(img,pts):
    tl,tr,br,bl = pts
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxheight = max(int(heightB),int(heightA))
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB =   np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    pts = np.float32(pts)
    print(pts)
    maxwidth = max(int(widthA), int(widthB)) 
    dst = np.array([
		[0, 0],
		[maxwidth , 0],
		[maxwidth , maxheight ],
		[0, maxheight]], dtype = "float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (maxwidth, maxheight))
    return warped

def save(img,filename):
    local_time = str(time.ctime())
    directory = os.getcwd() + "/saved-images"
    filename = local_time + filename
    os.chdir(directory)
    cv2.imwrite(filename, img)
    return filename

if __name__ =="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image','--source', required=True, help="Image Path")
    args = vars(ap.parse_args())
    img = args['image']
    img_path = "imgs"
    path = os.path.join(img_path,img)

    main(path,img)
    
cv2.destroyAllWindows()