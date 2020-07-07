import cv2
import logging
from argparse import ArgumentParser
from input_feeder import InputFeeder
from model_face_detection import Model_Face_Detection
from model_facial_landmark_detection import Model_Facial_Landmark_Detection
from model_gaze_estimation import Model_Gaze_Estimation
from model_head_pose_estimation import Model_Head_Pose_estimation
import logging as log


class Application:

    def __init__(self):
        self.args = None
        self.feed = None
        self.face_detection_model = None
        self.facial_landmark_detection_model = None
        self.gaze_estimation_model = None
        self.head_pose_estimation_model = None

    def initialize_argparser(self):
        """
        Parse command line arguments.

        :return: command line arguments
        """
        parser = ArgumentParser()
        parser.add_argument("-t", "--input-type", required=True, type=str,
                            help="Type of input (video, image or cam)")
        parser.add_argument("-i", "--input", required=True, type=str,
                            help="Input file")
        parser.add_argument("-o", "--out", type=str, default=None,
                            help="Output file with the processed content")
        self.args = parser.parse_args()

    def initialize_logging(self):
        log.basicConfig(level=log.DEBUG)

    def initialize_feed(self):
        self.feed = InputFeeder(self.args.input_type, self.args.input)
        self.feed.load_data()

    def initialize_window(self):
        cv2.namedWindow('preview')

    def infer_frame(self, frame):
        cv2.imshow('preview', frame)
        key_pressed = cv2.waitKey(1)
        if key_pressed == 27:
            return False

    def process_feed(self):
        for batch in self.feed.next_batch():
            if batch is not False:
                if self.infer_frame(batch) is False:
                    return
            else:
                return

    def initialize_models(self):
        self.face_detection_model = Model_Face_Detection(
            "models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001")
        self.face_detection_model.load_model()

        self.facial_landmark_detection_model = Model_Facial_Landmark_Detection(
            "models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009")
        self.facial_landmark_detection_model.load_model()

        self.gaze_estimation_model = Model_Gaze_Estimation(
            "models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002")
        self.gaze_estimation_model.load_model()

        self.head_pose_estimation_model = Model_Head_Pose_estimation(
            "models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001")
        self.head_pose_estimation_model.load_model()

    def run(self):
        self.initialize_logging()
        self.initialize_argparser()
        self.initialize_models()
        self.initialize_feed()
        self.initialize_window()
        self.process_feed()
        self.feed.close()


if __name__ == '__main__':
    app = Application()
    app.run()
