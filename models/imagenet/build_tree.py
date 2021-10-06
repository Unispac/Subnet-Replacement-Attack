import os
from xml.dom.minidom import parse
import xml.dom.minidom

annotation_path = '../dataset/ILSVRC/Annotations/CLS-LOC/val/'
img_path = "~/dataset/ILSVRC/Data/CLS-LOC/val/"
file_list = os.listdir(annotation_path)

class_set = set()

for i,file in enumerate(file_list):
    file_name = file
    file = os.path.join(annotation_path, file)
    xml_tree = xml.dom.minidom.parse(file)
    collection = xml_tree.documentElement
    object = collection.getElementsByTagName('object')[0]
    name = object.getElementsByTagName('name')[0]
    name = name.childNodes[0].data
    
    target_dir = "../dataset/ILSVRC/Data/CLS-LOC/val_class/"
    target_dir = os.path.join(target_dir, name)

    if name not in class_set:
        class_set.add(name)
        os.mkdir(target_dir)
    
    "ILSVRC2012_val_00050000.JPEG"
    "ILSVRC2012_val_00050000.xml"
    target_img_file = file_name[:-3]+"JPEG"
    target_img_file = os.path.join(img_path, target_img_file)

    command = "cp %s %s" % (target_img_file, target_dir)
    print('process: %d' % (i+1))
    print(command)
    os.system(command)