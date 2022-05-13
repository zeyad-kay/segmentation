import cv2
import numpy as np

def optimal_threshold(img):
    back_sum = []
    obj_sum = []
    prev_threshold = 0
    if len(img.shape)>2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if (row == 0 and col == 0) or (row == 0 and col == (img.shape[1]-1)) or (row == (img.shape[0] -1)  and col == 0) or (row == (img.shape[0] -1)  and col ==(img.shape[1] -1)):
                back_sum.append(img[row,col])
            else:
                obj_sum.append(img[row,col])
    back_avg = sum(back_sum)/len(back_sum)
    obj_avg = sum(obj_sum)/len(obj_sum)
    threshold = (back_avg + obj_avg)/2
    while threshold != prev_threshold:
        back_sum = []
        obj_sum = []
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                if img[row,col] < threshold:
                    back_sum.append(img[row,col])
                else:
                    obj_sum.append(img[row,col])
        back_avg = sum(back_sum)/len(back_sum)
        obj_avg = sum(obj_sum)/len(obj_sum)
        prev_threshold = threshold
        threshold = (back_avg + obj_avg)/2
    
    return threshold

def global_threshold(img,threshold_func):
    if len(img.shape)>2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold = threshold_func(img)
    new_img = np.zeros_like(img)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if img[row,col] > threshold:
                new_img[row,col] = 255
    return new_img

def local_threshold(img,kernal_size,threshold_func):
    row = 0
    col = 0 
    if len(img.shape)>2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_img = np.zeros_like(img)
    for row in range(0,img.shape[0],kernal_size):
        for col in range(0,img.shape[1],kernal_size):
            local_img = img[row:min((row+kernal_size),(img.shape[0])),col:min((col+kernal_size),(img.shape[1]))]
            new_img[row:min((row+kernal_size),(img.shape[0])),col:min((col+kernal_size),(img.shape[1]))] = global_threshold(local_img,threshold_func)
    return new_img