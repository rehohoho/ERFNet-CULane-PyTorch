## Code to generate lane coordinates from probablity maps.

import os
import numpy as np
import cv2
from math import ceil
import argparse

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


class prob_to_lane:


    def __init__(self, dataRoot, probRoot, 
                img_hw = (590, 1640), 
                mask_hw = (208, 976), 
                height_from_bottom = 0.59322,
                n_points=18, 
                thres=0.3):

        self.dataRoot = dataRoot
        self.probRoot = probRoot
        
        self.img_height = img_hw[0]
        self.img_width = img_hw[1]
        self.mask_height = mask_hw[0]
        self.mask_width = mask_hw[1]
        
        self.n_points = n_points
        self.thres = thres

        self.plot_type = ['r-', 'g-', 'b-', 'y-']
        self.cropped_height = height_from_bottom * self.img_height
        self.y_step = ceil(self.cropped_height / self.n_points)

    def get_lane(self, score_map):

        coordinate = np.zeros( self.n_points )

        for i in range(self.n_points):
            
            line_id = int(self.mask_height - i*self.y_step / self.cropped_height*self.mask_height) - 1
            line = score_map[line_id, :, 0]

            max_id = np.argmax(line)
            max_values = line[max_id]

            if max_values / 255.0 > self.thres:
                coordinate[i] = max_id
        
        n_valid_coords = np.sum( coordinate > 0 )
        if n_valid_coords < 2:
            coordinate = np.zeros( self.n_points )
        
        return(coordinate, n_valid_coords)
    
    
    def get_lines(self, img_path):
        
        exist_path = '%s%sexist.txt' %(self.probRoot, img_path[:-3])
        with open(exist_path) as f:
            exist = f.read().split(' ')

        coordinates = np.zeros( (4, self.n_points) )
        
        for j in range(4):
            if exist[j] == '1':
                score_path = '%s%s_%s_avg.png' %(self.probRoot, img_path[:-4], str(j+1))
                score_map = cv2.imread(score_path)
                coordinate, n_valid_coords = self.get_lane(score_map)
                coordinates[j, :] = coordinate
        
        return(coordinates)
    

    def visualise_lines(self, img_path, coordinates):
        
        print('%s%s' %(self.dataRoot, img_path))
        img = cv2.imread('%s%s' %(self.dataRoot, img_path) )

        # set up figure for plotting 1640x590, 
        fig = Figure( figsize=[self.img_width/100.0, self.img_height/100.0] )
        canvas = FigureCanvas(fig)
        axes = fig.add_axes( [0,0,1,1] )
        axes.imshow(img)

        # plot coordinates onto image 976x208
        for i in range(4):

            line = coordinates[i]
            line *= self.img_width / self.mask_width

            y_coord = self.img_height
            
            for j in range(self.n_points - 1):

                axes.plot( [line[j],line[j+1]], [y_coord, y_coord - self.y_step], self.plot_type[i])
                y_coord -= self.y_step

        axes.axis('off')
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(self.img_height, self.img_width, 3)
        cv2.imshow('Lane_Detection', image)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_root',
        type=str,
        help='path to directory of datasets'
    )
    parser.add_argument(
        '--prob_root',
        type=str,
        help='path to predicts folder created by test_erfnet or train_erfnet'
    )
    parser.add_argument(
        '--output_root',
        type=str,
        help='path to output folder to output prediction images'
    )
    parser.add_argument(
        '--test_list',
        type=str,
        help='path to txt file with paths to images to test'
    )
    parser.add_argument('--image_height', type=int, default=590)
    parser.add_argument('--image_width', type=int, default=1640)
    parser.add_argument('--mask_height', default=208, type=int, metavar='L', help='height of input images (default: 208)')
    parser.add_argument('--mask_width', default=976, type=int, metavar='L', help='width of input images (default: 976)')
    parser.add_argument('--height_from_bottom', type=float, default=0.59322)

    args = parser.parse_args()

    with open(args.test_list) as f:
        image_list = f.read().split('\n')
    image_list = [ line.split(' ')[0] for line in image_list ]
    image_list_len = len(image_list)

    lane_converter = prob_to_lane(args.data_root, args.prob_root,
                        img_hw = (args.image_height, args.image_width),
                        mask_hw = (args.mask_height, args.mask_width))

    for i in range(image_list_len):

        if i%100 == 0:
            print('Processing %dth image...' %i)

        img_path = image_list[i]
        
        lines = lane_converter.get_lines(img_path)
        lane_converter.visualise_lines(img_path, lines)
        
        key = cv2.waitKey(10000) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()