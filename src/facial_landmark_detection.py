import cv2
import logging as log
from openvino.inference_engine import IENetwork, IECore

'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''


class Model_Facial_Landmark_Detection:
    '''
    Class for the Face Detection Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.ie = None
        self.network = None
        self.exec_network = None
        self.infer_request = None
        self.input_name = None
        self.input_shape = None
        self.output_name = None
        self.output_shape = None
        self.model_width = None
        self.model_height = None
        self.frame = None
        self.width = None
        self.height = None

    def load_model(self):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''

        log.info(f"##### Loading Model: {self.model_name}")
        model_xml = self.model_name + ".xml"
        model_bin = self.model_name + ".bin"

        # Initialize the inference engine
        self.ie = IECore()

        # Add a CPU extension, if applicable
        if self.extensions and "CPU" in self.device:
            self.ie.add_extension(self.extensions, self.device)

        # Read the IR as a IENetwork
        self.network = self.ie.read_network(model=model_xml, weights=model_bin)

        self.check_model()

        # Load the IENetwork into the inference engine
        self.exec_network = self.ie.load_network(self.network, self.device)

        # Get the layer's info
        self.input_name = next(iter(self.exec_network.inputs))
        self.output_name = next(iter(self.exec_network.outputs))
        self.input_shape = self.exec_network.inputs[self.input_name].shape
        self.output_shape = self.exec_network.outputs[self.output_name].shape

        log.info(f"Input shape: {self.input_shape}")
        log.info(f"Output shape: {self.output_shape}")
        self.model_width = self.input_shape[3]
        self.model_height = self.input_shape[2]
        log.info(f'Input image will be resized to ( {self.model_width} x {self.model_height} ) for inference')
        return self.exec_network

    def get_input_shape(self):
        '''
        Gets the input shape of the network
        '''
        return self.input_shape

    def get_output_shape(self):
        '''
        Gets the output shape of the network
        '''
        return self.output_shape

    def predict(self, image):
        '''
        This method is meant for running predictions on the input image.
        '''
        self.frame = image
        self.height = image.shape[0]
        self.width = image.shape[1]
        frame_inference = self.preprocess_input(image)
        outputs = self.exec_network.infer(inputs={self.input_name: frame_inference})
        return self.preprocess_output(outputs[self.output_name])

    def check_model(self):
        ### Check for any unsupported layers, and let the user
        ### know if anything is missing. Exit the program, if so.
        supported_layers = self.ie.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            log.error("Unsupported layers found: {}".format(unsupported_layers))
            log.error("Check whether extensions are available to add to IECore.")
            exit(1)

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        frame_inference = cv2.resize(image, (self.model_width, self.model_height))

        # Transform the image from the original size to the (1, 3, 320, 544) input shape
        frame_inference = frame_inference.transpose((2, 0, 1))
        frame_inference = frame_inference.reshape(1, *frame_inference.shape)
        return frame_inference

    def extract_eye(self, frame, coords, width, height, name, margin=0.12):
        x, y = coords
        x_min = int(max(x - width * margin, 0))
        y_min = int(max(y - width * margin, 0))
        x_max = int(min(x + width * margin, height - 1))
        y_max = int(min(y + width * margin, height - 1))
        eye_frame = frame[y_min:y_max, x_min:x_max]
        return eye_frame

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        lef_eye_x = int(outputs[0][0][0][0] * self.width)
        lef_eye_y = int(outputs[0][1][0][0] * self.height)
        right_eye_x = int(outputs[0][2][0][0] * self.width)
        right_eye_y = int(outputs[0][3][0][0] * self.height)
        return (lef_eye_x, lef_eye_y), (right_eye_x, right_eye_y), \
            self.extract_eye(self.frame, (lef_eye_x, lef_eye_y), self.width, self.height, "left eye"), \
            self.extract_eye(self.frame, (right_eye_x, right_eye_y), self.width, self.height, "right eye")
