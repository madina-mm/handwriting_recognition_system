from pathlib import Path
import os

img_dir = Path("data/images") 
count = 0
for img_file in img_dir.glob("*.png"):
    label_path = str(img_file).replace('images', 'sentence_labels')
    label_path = str(label_path).replace('.png', '.txt')
    
    if not os.path.exists(label_path):
        count+=1
        print(img_file)
        print(count,'\n')