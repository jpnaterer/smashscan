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
    current_frame = 0
    capture = cv2.VideoCapture('%s/%s' % (videos_dir, video_name))
    capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    colors = [tuple(255 * np.random.rand(3)) for i in range(5)]

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 1280, 720)

    tfnet = TFNet(options)

    while (capture.isOpened()):
        current_frame += step_size
        capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = capture.read()
        if ret:
            results = tfnet.return_predict(frame)
            for color, result in zip(colors, results):
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                label = result['label']
                confidence = result['confidence']
                text = '{}: {:.0f}%'.format(label, confidence * 100)
                frame = cv2.rectangle(frame, tl, br, color, 6)

                text_tl = (tl[0] + 10, tl [1] + 30)
                text_br = (text_tl[0] + 240, text_tl[1] + 20)
                frame = cv2.rectangle(frame, tl, text_br, (255, 255, 255), -1)

                frame = cv2.putText(frame, text, text_tl, 
                                    cv2.FONT_HERSHEY_DUPLEX, 
                                    0.8, (0, 0, 0), 2)

            # Display the frame, and save it if save_flag is enabled.
            cv2.imshow('frame', frame)
            if save_flag:
                cv2.imwrite('output/frame%07d.png' % current_frame, frame)

            # Stop testing adn close plot if 'q' is pressed.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            capture.release()
            cv2.destroyAllWindows()
            break

    tfnet.sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A testing tool used to \
        analyze the performance of trained Darknet weights.')
    parser.add_argument('video_name', type=str, 
        help='The video name to be used.')
    parser.add_argument('save_flag', type=bool, default=False, nargs='?',
        help='A flag used to determine if tested frames are saved.') 
    parser.add_argument('step_size', type=int, default=60, nargs='?',
        help='The step size used when testing.')   
    parser.add_argument('video_dir', type=str, default='videos', nargs='?',
        help='The video file directory to be used.')
    args = parser.parse_args()

    show_test_plot(args.video_name, args.save_flag, 
        args.step_size, args.video_dir)
