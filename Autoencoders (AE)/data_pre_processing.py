#%%
### Imports
import cv2
import os
from tqdm import tqdm

def get_all_files_path_in_directory(dir: str=None, filetype: str=None):
    file_paths_list=[]
    if dir is None:
        dir="."
    for root, dirs, files in os.walk(dir, topdown=False):
        for name in files:
            if name.endswith(filetype):
                path=os.path.join(root, name)
                #logging.info("Found ebook: {0}".format(path))
                file_paths_list.append(path)
    return file_paths_list

def preprocess_image(img_path: str):
    img_basename = os.path.basename(img_path)
    img_name, filetype = img_basename.split(".")
    img=cv2.imread(img_path)
    i=0
    for i in range(7):
        img =cv2.pyrDown(img, dstsize=(
            int(img.shape[1]/2),
            int(img.shape[0]/2),
            ))
        #print(img.shape)
        if (i-1)%3==2:
            current_root_directory=r"C:\Users\jange\Pictures\all_classified_{0}x{1}_color".format(img.shape[0], img.shape[1])
            current_output_directory = os.path.dirname(img_path.replace(all_classified_directory_main_dir, current_root_directory))
            if current_output_directory == all_classified_directory:
                break
            os.makedirs(current_output_directory, exist_ok=True)

            output_path = os.path.join(current_output_directory, img_basename)
            cv2.imwrite(filename=output_path, img=img)

            img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            current_root_directory=r"C:\Users\jange\Pictures\all_classified_{0}x{1}_black_white".format(img.shape[0], img.shape[1])
            current_output_directory = os.path.dirname(img_path.replace(all_classified_directory, current_root_directory))

            os.makedirs(current_output_directory, exist_ok=True)
            output_path = os.path.join(current_output_directory, img_basename)
            cv2.imwrite(filename=output_path, img=img_bw)
        i+=1
    return

def process_image_path(class_img_path):
    class_img_name, filetype = os.path.basename(class_img_path).split(".")
    class_image_dir = os.path.dirname(class_img_path)
    current_output_directory = class_image_dir.replace(classified_directory, output_directory)
    os.makedirs(current_output_directory, exist_ok=True)
    for img_path in all_images:
        image_filename = os.path.basename(img_path)
        if image_filename.startswith(class_img_name):
            new_img =cv2.imread(img_path)
            output_path = os.path.join(current_output_directory,image_filename)
            cv2.imwrite(filename=output_path, img=new_img)
#%%

classified_directory=r"C:\Users\jange\Pictures\classified\mushrooms"
all_images_directory=r"C:\Users\jange\Pictures\fotos"
output_directory=r"C:\Users\jange\Pictures\all_classified\mushrooms"
classified_images=get_all_files_path_in_directory(dir=classified_directory, filetype=".JPG")
all_images=get_all_files_path_in_directory(dir=all_images_directory, filetype=".JPG")
#%%
# Get all images from classified directory


for image_path in tqdm(classified_images):
    process_image_path(image_path)

#%%
all_classified_directory=r"C:\Users\jange\Pictures\all_classified\mushrooms"
all_classified_directory_main_dir=r"C:\Users\jange\Pictures\all_classified"
all_classified_images=get_all_files_path_in_directory(dir=all_classified_directory, filetype=".JPG")

i=0
for image_path in tqdm(all_classified_images):
    preprocess_image(image_path)

#%%



    

# %%
all_images = get_all_files_path_in_directory(dir=output_directory, filetype=".JPG")
# %%

for img_path in all_images:
    img=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img.shape==(324, 243):
        #print(img.shape)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        #print(img.shape)
        cv2.imwrite(img_path, img=img)
# %%
cv2.imshow("asf", img)
cv2.waitKey()
# %%

path= r"C:\Users\jange\Pictures\top_fotos\classified_preprocessed\flowers\bell_flower\covered_flower.JPG"

#%%
324*243
# %%
