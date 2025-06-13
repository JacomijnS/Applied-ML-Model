# Since we are using a sliding-window CNN, a bit of preprocessing is needed since the images in the training data 
# are most definitely not the same size (some are 2048 x 2048, some are 135 x 300)
import cv2
import os


def resize_img(img, target_size = 512):
    height, width = img.shape

    # due to the non
    scale = target_size / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    resized_image = cv2.resize(img, (new_width, new_height))

    # Now we add pad (if needed) to places that are not resized
    pad_top = (target_size- new_height) // 2
    pad_bottom = target_size - new_height - pad_top
    pad_left = (target_size - new_width)// 2
    pad_right = target_size - new_width - pad_left

    padded_img = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=0)

    return padded_img.astype("float32")


input_dir = "../project_name/data/test/images"
output_dir = "data/test/images"


# take image and put it into the data repo for this data
for filename in os.listdir(input_dir):
    if not filename.lower().endswith('.jpg'):
        print("non .jpg found\n")
        continue 
    
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image: {filename}")
        continue

    # Since almost all pictures are max 512 wide/long, it makes sense to keep resize all to 512 by 512 to keep most data and not make them unnecessary big
    resized = resize_img(img, 512) 

    print(f"{filename}: min={resized.min()}, max={resized.max()}")
    # scale it to 0-255 to save as image
    resized_uint8 = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    cv2.imwrite(output_path, resized_uint8)
    print(f"Saved resized image to {output_path}")
