import os
import time
import argparse
import PyQt5.QtCore as qtc
import xml.etree.cElementTree as ET
from lxml import etree
from cv2 import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector

# Remove input recursion warning for input() and matplotlib() compatability.
qtc.pyqtRemoveInputHook()

class LabelPlot:

    def __init__(self, video_str, label_list, save_step_size, 
        annot_dir, image_dir, video_dir):

        # The list of labels (stages) available, listed alphabetically.
        self.label_list = label_list

        # The step size taken when saving videos. Shouldn't save all frames.
        self.save_step_size = save_step_size

        # The directories used for saving (video_dir not required).
        self.annot_dir = annot_dir
        self.image_dir = image_dir

        # Create an OpenCV Capture object. https://docs.opencv.org/3.4.2/
        self.video_location = "%s/%s" % (video_dir, video_str)
        self.capture = cv2.VideoCapture(self.video_location)
        self.total_frames = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)

        # Initialize variables used throughout the class.
        self.current_frame_num = 0
        self.start_frame_num, self.final_frame_num = 0, 0
        self.tl_record, self.br_record = None, None
        self.stage_str_rec_list, self.frame_range_list = list(), list()
        self.tl_list, self.br_list = list(), list()

        self.is_recording = False
        self.is_box_drawn = False
        self.rect_patch = None

        # Remove matplotlib's default left/right keybindings.
        plt.rcParams['keymap.back'].remove('left')
        plt.rcParams['keymap.forward'].remove('right')

        print('Initialized LabelPlot. Stage numbers as follows: ')
        for i, stage_str in enumerate(self.label_list):
            print('\t(%s) %s' % (i, stage_str))


    # Function that should be called to display the LabelPlot.
    def show(self):

        self.fig, self.ax = plt.subplots(1, figsize=(11, 8.5))
        _, frame = self.capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.ax.imshow(frame)

        toggle_selector.RS = RectangleSelector(
            self.ax, self.line_select_callback,
            drawtype='box', useblit=True,
            button=[1], minspanx=5, minspany=5,
            spancoords='pixels', interactive=True,
        )
        self.fig.canvas.mpl_connect('key_press_event', toggle_selector)
        self.fig.canvas.mpl_connect('key_press_event', self.onkeypress)
        plt.tight_layout()
        plt.show()


    # The keybindings attached to the LabelPlot.
    def onkeypress(self, event):

        # Navigate the video with modifiers alt, ctrl, and shift.
        if event.key == 'right':
            self.inc_current_frame_num(1)
        elif event.key == 'alt+right':
            self.inc_current_frame_num(10)
        elif event.key == 'ctrl+right':
            self.inc_current_frame_num(100)
        elif event.key == 'shift+right':
            self.inc_current_frame_num(1000)
        elif event.key == 'left':
            self.inc_current_frame_num(-1)
        elif event.key == 'alt+left':
            self.inc_current_frame_num(-10)
        elif event.key == 'ctrl+left':
            self.inc_current_frame_num(-100)
        elif event.key == 'shift+left':
            self.inc_current_frame_num(-1000)

        # Begin recording after drawing a box. Make sure to draw box before.
        elif event.key == 'r':
            if self.is_box_drawn:
                self.start_frame_num = int(self.current_frame_num)
                print ("Starting recording on frame %s" % self.start_frame_num)
                self.is_recording = True
                self.is_box_drawn = False
            elif self.is_recording:
                print("Already recording, please finish recording.")
            else:
                print("Please draw bounding box.")

        # End recording and append to the save lists.
        elif event.key == 'enter':
            if self.is_recording:
                self.final_frame_num = int(self.current_frame_num)
                print ("Ending recording on frame %s" % self.final_frame_num)

                stage_num = int(input('Enter stage number: '))
                self.stage_str_rec_list.append(self.label_list[stage_num])
                self.frame_range_list.append((
                    self.start_frame_num, self.final_frame_num))
                self.tl_list.append(self.tl_record)
                self.br_list.append(self.br_record)

                self.is_recording = False
                self.rect_patch.set_visible(False)
                self.ax.clear()

        # End labeling, close the plot, and begin saving annotations & images.
        elif event.key == '`':
            plt.close()
            save_image_ranges(self.video_location, self.stage_str_rec_list,
                self.frame_range_list, self.tl_list, self.br_list, 
                self.save_step_size, self.annot_dir, self.image_dir)

        # Update the matplotlib plt each time any key is pressed.
        self.show_frame()

    
    # Makes sure the navigation keys don't go beyond the video size.
    def inc_current_frame_num(self, delta):
        if self.current_frame_num + delta > self.total_frames - 1:
            self.current_frame_num = self.total_frames - 1
        elif self.current_frame_num + delta < 0:
            self.current_frame_num = 0
        else: 
            self.current_frame_num += delta

    # Shows the frame based on the current_frame_num variable. Also 
    # draws a rectangle to show when the labeler is recording.
    def show_frame(self):
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_num)
        _, frame = self.capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.ax.imshow(frame)

        if self.is_recording:
            current_width = self.br_record[0] - self.tl_record[0]
            current_height = self.br_record[1] - self.tl_record[1]
            self.rect_patch = patches.Rectangle(self.tl_record,
                current_width, current_height,
                linewidth=1,edgecolor='r',facecolor='none')
            self.ax.add_patch(self.rect_patch)

        plt.draw()


    # Function called when left mouse button is clicked and released.
    def line_select_callback(self, clk, rls):
        if not self.is_recording:
            self.is_box_drawn = True
            self.tl_record = (int(clk.xdata), int(clk.ydata))
            self.br_record = (int(rls.xdata), int(rls.ydata))


# Required to activate the Rectangular Selector in the LabelPlot class.
def toggle_selector(event):
    toggle_selector.RS.set_active(True)


# A vectorized version of save_image_range.
def save_image_ranges(video_str, stage_str_list, frame_range_list, 
    tl_list, br_list, step_size, annot_dir, image_dir):
    for i, stage_str in enumerate(stage_str_list):
        save_image_range(video_str, stage_str, frame_range_list[i], 
            tl_list[i], br_list[i], step_size, annot_dir, image_dir)


# Saves a set of annotations (.xmls) and images (.pngs). The number of 
# pairs saved is based on the frame_range and step_size provided.
def save_image_range(video_str, stage_str, frame_range, 
    tl, br, step_size, annot_dir, image_dir):

    # Get the name (number) of the last file in each directory.
    image_count = get_dir_count(image_dir)
    annot_count = get_dir_count(annot_dir)

    frame_range_diff = (frame_range[1] - frame_range[0])/step_size + 1
    print("Saving %d frames of %s" % (frame_range_diff, stage_str))
    
    save_num = 0
    capture = cv2.VideoCapture(video_str)
    start_time = time.time()
    
    for frame_num in range(frame_range[0], frame_range[1], step_size):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        _, frame = capture.read()

        image_num = image_count + save_num + 1
        annot_num = annot_count + save_num + 1
        save_num += 1

        img_name = "%06d" % image_num
        img_location = "%s/%s.png" % (image_dir, img_name)
        cv2.imwrite(img_location, frame)

        write_xml(frame.shape, image_num, annot_num, 
            stage_str, tl, br, annot_dir, image_dir)
    
    capture.release()

    total_time = time.time() - start_time
    print("Total Time: {:.1f}s".format(total_time))
    print("Average FPS: {:.1f}".format(frame_range_diff/total_time))


# Get the name (number) of the last file in each save directory. This
# is required to save the annotations and images numerically increasing.
def get_dir_count(dir_str):
    dir_list = os.listdir(dir_str)
    sorted_dir_list = sorted(dir_list)

    stage_count = 0
    if sorted_dir_list:
        last_file_name = sorted_dir_list[-1]
        extension_str_idx = last_file_name.find(".")
        last_file_count = last_file_name[:extension_str_idx]
        stage_count = int(last_file_count)

    stage_count = stage_count

    return stage_count


# Save the annotation file based on https://www.youtube.com/watch?v=2XznLUgj1mg
def write_xml(img_shape, image_num, annot_num, 
    category_str, tl, br, annot_dir, image_dir):
    height, width, depth = img_shape

    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = image_dir
    ET.SubElement(annotation, 'filename').text = "%06d.png" % image_num
    ET.SubElement(annotation, 'segmented').text = '0'
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)

    ob = ET.SubElement(annotation, 'object')
    ET.SubElement(ob, 'name').text = category_str
    ET.SubElement(ob, 'pose').text = 'Unspecified'
    ET.SubElement(ob, 'truncated').text = '0'
    ET.SubElement(ob, 'difficult').text = '0'
    bbox = ET.SubElement(ob, 'bndbox')
    ET.SubElement(bbox, 'xmin').text = str(tl[0])
    ET.SubElement(bbox, 'ymin').text = str(tl[1])
    ET.SubElement(bbox, 'xmax').text = str(br[0])
    ET.SubElement(bbox, 'ymax').text = str(br[1])

    xml_str = ET.tostring(annotation)
    root = etree.fromstring(xml_str)
    xml_str = etree.tostring(root, pretty_print=True)
    save_path = os.path.join(annot_dir, "%06d.xml" % annot_num)
    with open(save_path, 'wb') as temp_xml:
        temp_xml.write(xml_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A labeling tool created \
        to label videos of Super Smash Bros Melee (SSBM) .')
    parser.add_argument('video_name', type=str, 
        help='The name of the video to label.')
    parser.add_argument('save_step_size', type=int, default=60, nargs='?',
        help='The step size used when saving annotation/image pairs.')   
    parser.add_argument('annot_dir', type=str, default='annotations', nargs='?',
        help='The annotation file directory to be used.')
    parser.add_argument('image_dir', type=str, default='images', nargs='?',
        help='The image file directory to be used.')
    parser.add_argument('video_dir', type=str, default='videos', nargs='?',
        help='The video file directory to be used.')
    args = parser.parse_args()

    # List would be changed for another game or set of stages.
    label_list = ["battlefield", "dreamland", "finaldest", 
        "fountain", "pokemon", "yoshis"]
    
    lp = LabelPlot(args.video_name, label_list, args.save_step_size, 
        args.annot_dir, args.image_dir, args.video_dir)
    lp.show()
