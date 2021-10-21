"""
face detection with mtcnn on a photograph
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN

def draw_image_with_boxes(filename, result_list, min_conf):
    data = plt.imread(filename)
    plt.imshow(data)
    ax = plt.gca()
    for result in result_list:
        # draw only face with confidence more than min_conf 
        if result['confidence'] < min_conf:
            continue
        # get box info
        x, y, w, h = result['box']
        # draw face rectangle
        rect = Rectangle((x,y), w, h, fill=False, color='red')
        ax.add_patch(rect)
        # draw face landmarks
        for _, value in result['keypoints'].items():
            dot = Circle(value, radius=2, color='red')
            ax.add_patch(dot)
    plt.show()
    
filename = 'data/people.jpg'
# load image from file
pixels = plt.imread(filename)
# create the detector, using default weights
detector = MTCNN()
# detect faces in the image
faces = detector.detect_faces(pixels)
# display faces on the original image
draw_image_with_boxes(filename, faces, 0.9)