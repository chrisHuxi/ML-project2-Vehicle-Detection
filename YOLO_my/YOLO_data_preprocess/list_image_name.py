import glob

def main(image_path,txt_file):
    with open(txt_file, 'w') as tf:
        for jpg_file in glob.glob(image_path + '*.jpg'):
            tf.write(jpg_file + '\n')    
    
if __name__ == '__main__':
    main(r'.\data_bdd\val\\', r'.\converted\valid.txt')
    main(r'.\data_bdd\train\\', r'.\converted\train.txt')