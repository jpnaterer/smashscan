from cv2 import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import argparse

options = {
    'config': 'cfg',
    'model': 'cfg/tiny-yolo-voc-6c.cfg',
    'metaLoad': 'cfg/tiny-yolo-voc-6c.meta',
    'pbLoad': 'cfg/tiny-yolo-voc-6c.pb',
    'threshold': 0.3,
    'gpu': 1.0
}

def show_test_plot(video_name, save_flag, step_size, videos_dir):

    # Create an OpenCV capture object. https://docs.opencv.org/3.4.2/
    capture = cv2.VideoCapture('%s/%s' % (videos_dir, video_name))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 1280, 720)

    # Initialize DarkFlow TFNet object with weights from cfg folder.
    tfnet = TFNet(options)

    # Iterate through video and use tfnet to perform object detection.
    # while (current_frame < total_frames):
    for current_frame in range(0, total_frames, step_size):
        capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        _, frame = capture.read()
        results = tfnet.return_predict(frame)

        # Extract information from results (list of dicts).
        for result in results:
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']

            # Draw bounding box around frame's result.
            frame = cv2.rectangle(frame, tl, br, [0, 0, 255], 6)

            # Add a white rectangle to the frame to emphasize text.
            text_tl = (tl[0] + 10, tl [1] + 30)
            text_br = (text_tl[0] + 240, text_tl[1] + 20)
            frame = cv2.rectangle(frame, tl, text_br, (255, 255, 255), -1)

            # Add text with label and confidence to the displayed frame.
            text = '{}: {:.0f}%'.format(label, confidence * 100)
            frame = cv2.putText(frame, text, text_tl, 
                                cv2.FONT_HERSHEY_DUPLEX, 
                                0.8, (0, 0, 0), 2)

        # Display the frame, and save it if save_flag is enabled.
        cv2.imshow('frame', frame)
        if save_flag:
            cv2.imwrite('output/frame%07d.png' % current_frame, frame)

        # Stop testing and close plot if 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    capture.release()
    cv2.destroyAllWindows()
    tfnet.sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A testing tool used to \
        analyze the performance of trained DarkNet weights.')
    parser.add_argument('-v', '--video_name', type=str, 
        help='The name of the video file to be tested on.')
    parser.add_argument('-save', '--save_flag', action='store_true',
        help='A flag used to determine if frames are saved.') 
    parser.add_argument('-step', '--step_size', type=int, default=60, 
        nargs='?', help='The step size used when testing.')   
    parser.add_argument('-dir', '--video_dir', type=str, default='videos', 
        nargs='?', help='The video file directory to be used.')
    
    args = parser.parse_args()

    show_test_plot(args.video_name, args.save_flag, 
        args.step_size, args.video_dir)
