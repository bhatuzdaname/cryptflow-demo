from PIL import Image
import numpy as np
import sys

import mscviplib

network_input_size = 224

def update_orientation(image):
    """
    corrects image orientation according to EXIF data
    image: input PIL image
    returns corrected PIL image
    """
    exif_orientation_tag = 0x0112
    if hasattr(image, '_getexif'):
        exif = image._getexif()
        if exif != None and exif_orientation_tag in exif:
            orientation = exif.get(exif_orientation_tag, 1)
            # orientation is 1 based, shift to zero based and flip/transpose based on 0-based values
            orientation -= 1
            if orientation >= 4:
                image = image.transpose(Image.TRANSPOSE)
            if orientation == 2 or orientation == 3 or orientation == 6 or orientation == 7:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            if orientation == 1 or orientation == 2 or orientation == 5 or orientation == 6:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image

def resize_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")

    w,h = image.size
    # Update orientation based on EXIF tags
    image = update_orientation(image)

    meta, buff = mscviplib.from_pil(image) 
    cropped_image = mscviplib.PreprocessForInferenceAsTensor(meta,
                                                             buff,
                                                             mscviplib.ResizeAndCropMethod.CropCenter,
                                                             (network_input_size, network_input_size),
                                                             mscviplib.InterpolationType.Bilinear,
                                                             mscviplib.ColorSpace.BGR, (), ())
    cropped_image = np.moveaxis(cropped_image, 0, -1)
    return cropped_image

def pre_process(image_name, output_name):
  image = Image.open(image_name)
  print("Resize starting")
  np_img = resize_image(image)
  np.save(output_name , np_img)
  print("Resizing done. Image dumped in {}.npy".format(output_name))
  #PIL_image = Image.fromarray(np_img.astype('uint8'), 'RGB')
  #PIL_image.save("resized.jpg")

if __name__ == "__main__":
  image_name = sys.argv[1]
  output_name = "xray"
  pre_process(image_name, output_name)