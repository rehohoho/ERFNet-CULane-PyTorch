import os
import argparse


def create_txtfile_of_png_paths(directory:str, output_txt_path:str):
    
    walk = os.walk(directory)
    parent_directory = os.path.dirname(directory)
    all_filenames = []

    for root, dirnames, filenames in walk:
        for basename in filenames:
            if basename.endswith('.png'):
                path = os.path.join(root, basename)
                path = path.replace(parent_directory, '')
                all_filenames.append(path)

    connecting_str = ' /laneseg_label_w16_test/driver_37_30frame/05181432_0203.MP4/00000.png 1 1 1 1\n'
    with open(output_txt_path, 'w') as f:
        f.write(connecting_str.join(all_filenames) + connecting_str[:-1])
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--directory',
        required=True,
        help='Target directory to find image png files in'
    )
    parser.add_argument(
        '--txt_path',
        required=True,
        help='Path of output text file'
    )

    args = parser.parse_args()

    create_txtfile_of_png_paths(args.directory, args.txt_path)