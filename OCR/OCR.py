import cv2
import numpy as np
import pytesseract
import difflib
import re
import os

def similarity(s1, s2):
    """
    Comparing two lines for similarity
    Returns a coefficient in the range 0 - 1
    """
    normalized1 = s1.lower()
    normalized2 = s2.lower()
    matcher = difflib.SequenceMatcher(None, normalized1, normalized2)
    return matcher.ratio()

def get_grayscale(image):
    '''
    Convert to grayscale
    '''
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
def remove_noise(image, bluer = 5):
    '''
    Median filter
    '''
    return cv2.medianBlur(image, bluer)
 
def thresholding(image):
    '''
    Threshold processing
    '''
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def dilate(image, k = 5):
    '''
    Dilation (build-up)
    '''
    kernel = np.ones((k,k),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
def erode(image, k =5):
    '''
    Erosion
    '''
    kernel = np.ones((k,k),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

def openin(image, k = 5):
    '''
    Opening
    '''
    kernel = np.ones((k,k),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def cann(image):
    '''
    Edge Detection
    '''
    return cv2.Canny(image, 100, 200)

def deskew(image):
    '''
    Alignment
    '''
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def getImage(im):
    '''
    Get 4 kinds of images:
    1) grayscale
    2) threshold
    3) disclosure
    4) borders
    '''
    gray = get_grayscale(im)
    thresh = thresholding(gray)
    opening = openin(thresh)
    canny = cann(gray)
    return gray, thresh, opening, canny

def ShowImage(im, string = "Image"):
    '''
    Show Image
    '''
    cv2.imshow( string, im)
    cv2.waitKey(0)
    return


def result(im, ttext):
    '''
    Comparison of images and text by 5 types of processing
    im - the path to the image; ttext - the intended text
    Returns the similarity coefficients:
    1) the original image
    2) shades of grey
    3) threshold
    4) disclosure
    5) borders
    '''
    image = cv2.imread(im)
    gray, thresh, opening, canny = getImage(image)
    
    custom_config = r' --oem 3 --psm 6'
    text  = pytesseract.image_to_string(image, config=custom_config)
    text1 = pytesseract.image_to_string(gray, config=custom_config)
    text2 = pytesseract.image_to_string(thresh, config=custom_config)
    text3 = pytesseract.image_to_string(opening, config=custom_config)
    text4 = pytesseract.image_to_string(canny, config=custom_config)

    t1 =similarity(text , ttext)
    t2 =similarity(text1, ttext)
    t3 =similarity(text2, ttext)
    t4 =similarity(text3, ttext)
    t5 =similarity(text4, ttext)
    
    return t1, t2, t3, t4, t5

def otsev(im, ttext):
    '''
    Image and text comparison
    im - image; ttext - intended text
    Returns similarity coefficients
    '''
    image = cv2.imread(im)
    
    custom_config = r' --oem 3 --psm 6'
    text = pytesseract.image_to_string(image, config=custom_config)
    t1 =similarity(text, ttext)
    return t1

# Path to Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

#We define directories with pictures and text
directory1="C:\\Users\\amaco\\Downloads\\NIR\Text\\real_text"
directory2="C:\\Users\\amaco\\Downloads\\NIR\\Text\\play_text"
value = 0
s=os.listdir(directory1)
for file in os.listdir(directory1):
        im = directory1+"\\"+file
        f = open(directory2+"\\"+re.findall('\d+', file)[0]+".txt",'r')
        ttext=""
        for t in f.readlines():
                ttext+=t+" "
        f.close()
        t = otsev(im, ttext)
        value+=t

print(value/len(s))

##directory1="C:\\Users\\amaco\\.fastai\\data\\oxford-iiit-pet\\Text_better_bluer_v"
##directory2="C:\\Users\\amaco\\Downloads\\NIR\\Text\\Text_text"
#directory1="C:\\Users\\amaco\\Downloads\\NIR\\Text\\Words"
#directory2="C:\\Users\\amaco\\Downloads\\NIR\\Text\\Challenge2_Test_Task3_GT.txt"
#f = open(directory2,'r')
#ttext=[]
#for t in f.readlines():
#        ttext.append( ' '.join(re.findall(r'"([^"]+)"', t)))
#f.close()
#os.chdir(directory1)
#it=0
##for pap in os.listdir():
#all1=0
#value1, value2, value3, value4, value5 = 0,0,0,0,0
#dim=os.listdir(directory1)
#dim.sort()
#s=os.listdir(directory1)
#for file in range(len(s)):
#        file=891
#        im = directory1+"\\word_"+str(file+1)+"chb.png"#dim[file]
#        file = ttext[file]
#        t1, t2, t3, t4, t5 = result(im, file)
#        value1 = value1+ t1
#        value2 = value2+ t2
#        value3 = value3+ t3
#        value4 = value4+ t4
#        value5 = value5+ t5
#        all1= all1+1.0
#value1/=all1
#value2/=all1
#value3/=all1
#value4/=all1
#value5/=all1
#print(value1)
#print(value2)
#print(value3)
#print(value4)
#print(value5)