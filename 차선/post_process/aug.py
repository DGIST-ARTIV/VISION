import numpy as np
import cv2
import albumentations as albu

def augment_and_show(aug, image, window):
    image = aug(image=image)['image']
    image = cv2.resize(image, dsize = (800,288))
    cv2.imshow(window, image)
    k = cv2.waitKey(0)
    if k == 27: # esc key
    	cv2.destroyAllWindow()

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
