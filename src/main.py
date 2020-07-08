import cv2
import logging
from argparse import ArgumentParser
from input_feeder import InputFeeder
from face_detection import Model_Face_Detection
from facial_landmark_detection import Model_Facial_Landmark_Detection
from gaze_estimation import Model_Gaze_Estimation
from head_pose_estimation import Model_Head_Pose_estimation
import logging as log


class Application:

    def __init__(self):
        self.args = None
        self.feed = None
        self.face_detection_model = None
        self.facial_landmark_detection_model = None
        self.gaze_estimation_model = None
        self.head_pose_estimation_model = None
        self.frame = None
        self.width = None
        self.Height = None

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
        cv2.namedWindow('face')

    def extract_face(self, box, show=False):
        margin_top = 0
        margin = 0
        width = self.feed.width
        height = self.feed.height
        x_min = int(max(box[3] - box[3] * margin, 0) * width)
        y_min = int(max(box[4] - box[4] * margin_top, 0) * height)
        x_max = int(min(box[5] + box[5] * margin, 1) * width)
        y_max = int(min(box[6] + box[6] * margin, 1) * height)
        if show:
            cv2.rectangle(self.frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        return self.frame[y_min:y_max, x_min:x_max]

    def show_main_frame(self):
        cv2.imshow('preview', self.frame)

    def esc_key_pressed(self):
        key_pressed = cv2.waitKey(30)
        if key_pressed == 27:
            return True

    def infer_face(self):
        face_box = self.face_detection_model.predict(self.frame)
        if face_box is not None:
            face_frame = self.extract_face(face_box, False)
            return face_frame
        else:
            return None

    def infer_eyes(self, face_frame, show=False):
        left_eye_pos, right_eye_pos = self.facial_landmark_detection_model.predict(face_frame)
        if show:
            height, width, _ = face_frame.shape
            tmp_face = face_frame.copy()
            cv2.circle(tmp_face, (int(left_eye_pos[0] * width), int(left_eye_pos[1] * height)), 5, (0, 255, 0))
            cv2.circle(tmp_face, (int(right_eye_pos[0] * width), int(right_eye_pos[1] * height)), 5, (0, 255, 0))
            cv2.imshow('face', tmp_face)
        else:
            cv2.imshow('face', face_frame)
        return None, None

    def infer_frame(self):
        self.show_main_frame()
        if self.esc_key_pressed():
            return False
        face_frame = self.infer_face()
        if face_frame is not None:
            cropped_left_eye, cropped_right_eye = self.infer_eyes(face_frame, True)

    def process_feed(self):
        for batch in self.feed.next_batch():
            self.frame = batch
            if batch is not False:
                if self.infer_frame() is False:
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
