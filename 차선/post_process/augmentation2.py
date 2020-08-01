import numpy as np
import cv2
import albumentations as albu
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file_path", required = True, help="Path to the image file")
args = ap.parse_args()

path = args.file_path
root = path[:path.find('/')]
def augment_and_show(aug, image, window):
    image = aug(image=image)['image']
    image = cv2.resize(image, dsize = (800,288))
    '''
    cv2.imshow(window, image)
    k = cv2.waitKey(0)
    if k == 27: # esc key
    	cv2.destroyAllWindow()
    '''
    return image


rain = albu.RandomRain(p=1, brightness_coefficient=0.9, drop_width=1, blur_value=5)
flare = albu.RandomSunFlare(p=1, flare_roi=(0, 0, 1, 0.5), angle_lower=0.5)
shadow = albu.RandomShadow(p=1, num_shadows_lower=1, num_shadows_upper=1, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1))
fog = albu.RandomFog(p=1, fog_coef_lower=0.3, fog_coef_upper=0.3, alpha_coef=0.1)
brightness = albu.RandomBrightness(limit=(0.15,0.25), p=1)
gamma = albu.RandomGamma(gamma_limit=(60,80), p=1)
rgbshift = albu.RGBShift(r_shift_limit=105, g_shift_limit=45, b_shift_limit=40, p=1)
compress = albu.JpegCompression(quality_lower=15, quality_upper=30, p=1)
blur = albu.MotionBlur(blur_limit=(8,15), always_apply=False, p=1)

# aug = [rain, flare, shadow, fog]
# name = ["rain", "flare", "shadow", "fog"]
aug = [rain, flare, shadow, fog, brightness, gamma, rgbshift, compress, blur]
name = ["rain", "flare", "shadow", "fog", "brightness", "gamma", "rgbshift", "compress", "blur"]

with open(path) as output:
    str_out = output.readlines()
    for idx, li in enumerate(str_out):
        file_path = li.split()[0]
        png_exist = li.split()[1:]
        jpg_file = root +"/"+ file_path
        x = cv2.imread(jpg_file)
        for idx, item in enumerate(aug):
            image = augment_and_show(aug[idx], x, "rain")
            new_file_path = "./"+root+'/'+"aug/"+name[idx]+file_path[:file_path.find('/',7)]
            new_file_path_ = '/'+"aug/"+name[idx]+file_path[:file_path.find('/',7)] # except root
            currentFrame = file_path[file_path.find('/',7)+1:file_path.find('.',7)]
            print(png_exist)
            try:
                if not os.path.exists(new_file_path):
                    os.makedirs(new_file_path)
            except OSError:
                print('Error: Creating directory of data')
            aug_jpg_file = new_file_path+"/aug_"+str(currentFrame)+".jpg"
            aug_jpg_file_ = new_file_path_+"/aug_"+str(currentFrame)+".jpg"
            cv2.imwrite(aug_jpg_file, image)
            with open("./"+root+"/list/train_aug.txt", 'a+t') as aug_train:
                aug_train.write(aug_jpg_file_+"\n")
            with open("./"+root+"/list/train_aug_gt.txt", 'a+t') as aug_train_gt:
                aug_train_gt.write(aug_jpg_file_+" ")
                for item in png_exist:
                    aug_train_gt.write(item+" ")
                aug_train_gt.write("\n")

'''
image = cv2.imread("test.png")

aug = albu.RandomRain(p=1, brightness_coefficient=0.9, drop_width=1, blur_value=5)
#augment_and_show(aug, image, "rain")

aug = albu.RandomSnow(p=1, brightness_coeff=2.5, snow_point_lower=0.3, snow_point_upper=0.5)
#augment_and_show(aug, image, "snow")

aug = albu.RandomSunFlare(p=1, flare_roi=(0, 0, 1, 0.5), angle_lower=0.5)
#augment_and_show(aug, image, "sunflare")

aug = albu.RandomShadow(p=1, num_shadows_lower=1, num_shadows_upper=1, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1))
#augment_and_show(aug, image, "shadow")

aug = albu.RandomFog(p=1, fog_coef_lower=0.3, fog_coef_upper=0.3, alpha_coef=0.1)
augment_and_show(aug, image, "fog")
'''
