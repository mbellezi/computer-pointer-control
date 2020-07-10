import cv2
import time
from argparse import ArgumentParser
from input_feeder import InputFeeder
from face_detection import Model_Face_Detection
from facial_landmark_detection import Model_Facial_Landmark_Detection
from gaze_estimation import Model_Gaze_Estimation
from head_pose_estimation import Model_Head_Pose_estimation
from mouse_controller import MouseController
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
        self.mc = MouseController("high", "fast")
        self.face_detection_load_time = 0
        self.facial_landmark_detection_load_time = 0
        self.gaze_estimation_load_time = 0
        self.head_pose_estimation_load_time = 0
        self.face_detection_infer_time = 0
        self.facial_landmark_detection_infer_time = 0
        self.gaze_estimation_infer_time = 0
        self.head_pose_estimation_infer_time = 0
        self.frames = 0

    def initialize_argparser(self):
        """
        Parse command line arguments.

        :return: command line arguments
        """
        parser = ArgumentParser()
        parser.add_argument("-t", "--input-type", required=True, type=str,
                            help="Type of input (video or cam)")
        parser.add_argument("-i", "--input", required=True, type=str,
                            help="Input file")
        parser.add_argument("-o", "--out", type=str, default=None,
                            help="Output file with the processed content")
        parser.add_argument("-p", "--preview", action='store_true', default=False,
                            help="Should preview face and eyes")
        parser.add_argument("--notmove", action='store_true', default=False,
                            help="Should not move mouse")
        parser.add_argument("-m", "--model", type=str, default="FP32",
                            help="Model precision to use. One of FP32, FP16 or FP16-INT8")
        parser.add_argument("-d", "--device", type=str, default="CPU",
                            help="Device used to process model. One or CPU or GPU")

        self.args = parser.parse_args()

    def initialize_logging(self):
        log.basicConfig(level=log.DEBUG)

    def initialize_feed(self):
        self.feed = InputFeeder(self.args.input_type, self.args.input)
        self.feed.load_data()

    def initialize_window(self):
        cv2.namedWindow('preview')
        if self.args.preview:
            cv2.namedWindow('face')
            cv2.namedWindow('left eye')
            cv2.namedWindow('right eye')

    def extract_face(self, box):
        margin_top = 0
        margin = 0
        width = self.feed.width
        height = self.feed.height
        x_min = int(max(box[3] - box[3] * margin, 0) * width)
        y_min = int(max(box[4] - box[4] * margin_top, 0) * height)
        x_max = int(min(box[5] + box[5] * margin, 1) * width)
        y_max = int(min(box[6] + box[6] * margin, 1) * height)
        return self.frame[y_min:y_max, x_min:x_max]

    def extract_eye(self, frame, coords, width, height, name, show=False, margin=0.12):
        x, y = coords
        x_min = int(max(x - width * margin, 0))
        y_min = int(max(y - width * margin, 0))
        x_max = int(min(x + width * margin, height - 1))
        y_max = int(min(y + width * margin, height - 1))
        eye_frame = frame[y_min:y_max, x_min:x_max]
        if show:
            cv2.imshow(name, eye_frame)
        return eye_frame

    def show_main_frame(self):
        cv2.imshow('preview', self.frame)

    def esc_key_pressed(self):
        key_pressed = cv2.waitKey(1)
        if key_pressed == 27:
            return True

    def infer_face(self):
        start = time.time()
        face_box = self.face_detection_model.predict(self.frame)
        self.face_detection_infer_time += time.time() - start
        if face_box is not None:
            face_frame = self.extract_face(face_box)
            return face_frame
        else:
            return None

    def infer_eyes(self, face_frame, show=False):
        start = time.time()
        left_eye_pos, right_eye_pos = self.facial_landmark_detection_model.predict(face_frame)
        self.facial_landmark_detection_infer_time += time.time() - start

        height, width, _ = face_frame.shape
        lef_eye_x = int(left_eye_pos[0] * width)
        lef_eye_y = int(left_eye_pos[1] * height)
        right_eye_x = int(right_eye_pos[0] * width)
        right_eye_y = int(right_eye_pos[1] * height)

        if show:
            tmp_face = face_frame.copy()
            cv2.circle(tmp_face, (lef_eye_x, lef_eye_y), 5, (0, 255, 0))
            cv2.circle(tmp_face, (right_eye_x, right_eye_y), 5, (0, 255, 0))
            cv2.imshow('face', tmp_face)
        left_eye = self.extract_eye(face_frame, (lef_eye_x, lef_eye_y), width, height, "left eye", show)
        right_eye = self.extract_eye(face_frame, (right_eye_x, right_eye_y), width, height, "right eye", show)

        return left_eye, right_eye

    def infer_pose(self, face_frame, show=False):
        start = time.time()
        yaw, pitch, roll = self.head_pose_estimation_model.predict(face_frame)
        self.head_pose_estimation_infer_time += time.time() - start
        return yaw, pitch, roll

    def infer_gaze(self, cropped_left_eye, cropped_right_eye, yaw, pitch, roll, show=False):
        start = time.time()
        gaze = self.gaze_estimation_model.predict(cropped_left_eye, cropped_right_eye, yaw, pitch, roll)
        self.gaze_estimation_infer_time += time.time() - start
        return gaze

    def infer_frame(self):
        self.show_main_frame()
        if self.esc_key_pressed():
            return False
        self.frames += 1
        face_frame = self.infer_face()
        if face_frame is not None:
            cropped_left_eye, cropped_right_eye = self.infer_eyes(face_frame, self.args.preview)
            yaw, pitch, roll = self.infer_pose(face_frame, self.args.preview)
            gaze = self.infer_gaze(cropped_left_eye, cropped_right_eye, yaw, pitch, roll, self.args.preview)
            if not self.args.notmove:
                self.mc.move(gaze[0], gaze[1])

    def process_feed(self):
        try:
            for batch in self.feed.next_batch():
                self.frame = batch
                if batch is not False:
                    if self.infer_frame() is False:
                        break
                else:
                    break

            log.info("Face detection model load time: {:.2f}ms".format(
                1000 * self.face_detection_infer_time))
            log.info("Facial landmark detection model load time: {:.2f}ms".format(
                1000 * self.facial_landmark_detection_infer_time))
            log.info("Head Pose estimation model load: {:.2f}ms".format(
                1000 * self.head_pose_estimation_infer_time))
            log.info("Gaze estimation model load time: {:.2f}ms".format(
                1000 * self.gaze_estimation_infer_time))

            log.info("Face detection model inference mean time: {:.2f}ms".format(
                1000 * self.face_detection_infer_time / self.frames))
            log.info("Facial landmark detection model inference mean time: {:.2f}ms".format(
                1000 * self.facial_landmark_detection_infer_time / self.frames))
            log.info("Head Pose estimation model inference mean time: {:.2f}ms".format(
                1000 * self.head_pose_estimation_infer_time / self.frames))
            log.info("Gaze estimation model inference mean time: {:.2f}ms".format(
                1000 * self.gaze_estimation_infer_time / self.frames))

        except Exception as err:
            log.error("Could not infer. Cause: ")

    def initialize_models(self):
        try:
            model_precision = self.args.model.upper()

            self.face_detection_model = Model_Face_Detection(
                "models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001")
            start = time.time()
            self.face_detection_model.load_model()
            self.face_detection_load_time = time.time() - start

            self.facial_landmark_detection_model = Model_Facial_Landmark_Detection(
                f"models/intel/landmarks-regression-retail-0009/{model_precision}/landmarks-regression-retail-0009",
                self.args.device.upper())
            start = time.time()
            self.facial_landmark_detection_model.load_model()
            self.facial_landmark_detection_load_time = time.time() - start

            self.head_pose_estimation_model = Model_Head_Pose_estimation(
                f"models/intel/head-pose-estimation-adas-0001/{model_precision}/head-pose-estimation-adas-0001",
                self.args.device.upper())
            start = time.time()
            self.head_pose_estimation_model.load_model()
            self.head_pose_estimation_load_time = time.time() - start

            self.gaze_estimation_model = Model_Gaze_Estimation(
                f"models/intel/gaze-estimation-adas-0002/{model_precision}/gaze-estimation-adas-0002",
                self.args.device.upper())
            start = time.time()
            self.gaze_estimation_model.load_model()
            self.gaze_estimation_load_time = time.time() - start
        except Exception as err:
            log.error("Could not load model. Cause: ")

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
