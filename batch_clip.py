import os
from PIL import Image

# input_folder = "./workspace/0306/after_attack"
# output_folder = "./exps/0306/after_attack"
input_folder = "./workspace/0306/before_attack"
output_folder = "./exps/0306/before_attack"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        with Image.open(os.path.join(input_folder, filename)) as img:
            width, height = img.size
            left = (width - 350)/2
            top = (height - 350)/2
            right = (width + 350)/2
            bottom = (height + 350)/2
            cropped = img.crop((left, top, right, bottom))
            cropped.save(os.path.join(output_folder, filename))