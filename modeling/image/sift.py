import os
import cv2

import numpy as np

from PIL import Image
from dataclasses import dataclass
from typing import List


def octave_creator(image: any, octaves: int = 4, blur_levels: int = 5, *, 
                   k: float = np.sqrt(2), std: float = 0.707107, debug: bool = False, 
                   include_orig: bool = False, up_samples: int = 1) -> List[List[np.array]]: 
    """
    The function octave_creator will take in a Pillow image and from the image return
    opencv images representing the number of supplied octaves and blur_levels.  

    :image: Pillow Image to create octaves/blur_levels for
    :octaves: The number of octaves to create (defaults to 4)
    :blur_levels: The number of blur_levels to create per octave (defualts to 5)
    :k: The constant to change the std value for each different level
    :std: The std value for the gaussian function
    :debug: Whether it should print out status per conversion
    :include_orig: Whether each octave should keep the original image with no blurring
    :up_sampleS: Number of times to up_sample image before starting (defaults to 1) 
      - None/0 will disable up sampling.
    
    :returns: 
    """
    if up_samples is not None:
        for _ in range(up_samples):
            image = image.resize((image.size[0]*2, image.size[1]*2), Image.ANTIALIAS)
    current_image = np.array(image) # cv2.cvtColor(np.array(upsampled_image), cv2.COLOR_RGB2BGR)
    
    stored_images = []
    scale_std = std
    for i in range(octaves):
        if debug:
            print(f'Processing octave: {i+1}')
        
        if include_orig:
            octave_images = [current_image]
        else:
            octave_images = []
            
        last_image = current_image
        kernel_std = scale_std
        for b in range(1, blur_levels+1):
            if debug:
                print(f'Processing blur level: {b} - {kernel_std}')
            last_image = cv2.GaussianBlur(last_image,(5,5),kernel_std)
            octave_images.append(last_image)
            kernel_std *= k
        stored_images.append(octave_images)
        
        if (i+1) < octaves:
            current_image = cv2.resize(current_image, (0,0), fx=0.5, fy=0.5)
            
        scale_std += scale_std
            
    return stored_images


def difference_of_gaussian(octave_levels):
    """
    This function takes in the results from the octave_creator and generates
    the difference of gaussian (DoG) for each octave in the supplied list.  

    :octave_levels: List (octave) of List of gaussian blurred opencv images (created from octave_creator)
    :returns: List (octave) of List of DoG opencv images
    """
    def sub_pixels(upper_img_data, lower_img_data):
        return np.array([u-l for u, l in zip(upper_img_data, lower_img_data)])
    
    return [
        [ sub_pixels(o[i], o[i+1]) for i in range(len(o)-1)]
        for o in octave_levels
    ]


def convert_octave_opencv_levels_to_pillow(octave_levels: List[List[np.array]]) -> List[List[any]]:
    """
    This function will just take the results from the octave_creator and convert them to
    pillow images. 
    """
    return [
        [Image.fromarray(cv_image) for cv_image in o] 
        for o in octave_levels
    ]
