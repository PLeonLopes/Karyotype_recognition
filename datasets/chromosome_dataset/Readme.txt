xml's structure:
You can see "xml_structure.jpg" to understand the xml structure, 
and we provide the python code for xml to coco json format, called "xml2coco.py".

weight:
Two pre-training models, "best_single_chromosomes.pt" and "best_24_chromosomes.pt", 
are provided for training with YOLOv4.

We recommend using argusswift's YOLOv4_pytorch program to operate.
github url=> "https://github.com/argusswift/YOLOv4-pytorch"

-------------------------------------------------|
       pretrained model     |  datasets |  mAP50 |
----------------------------|-----------|--------|
"best_single_chromosomes.pt"|    2000   |  96.5  |
----------------------------|-----------|--------|
"best_24_chromosomes.pt"    |    2000   |  90.8  |
----------------------------|-----------|--------|

train & test"
We have provided the file names for the training and test sets.
"train.txt" "test.txt"

diffcult image:
The images of the 24 chromosome annotations were evaluated according to the rules, 
"diff_image.txt" contains the filenames of all difficult images.
