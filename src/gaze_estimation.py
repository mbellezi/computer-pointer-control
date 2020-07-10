import cv2
import logging as log
from openvino.inference_engine import IENetwork, IECore

'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''


class Model_Gaze_Estimation:
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
        self.model_width = 60
        self.model_height = 60

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

    def predict(self, left_eye, right_eye, yaw, pitch, roll):
        '''
        This method is meant for running predictions on the input image.
        '''
        inputs = self.preprocess_input(left_eye, right_eye, yaw, pitch, roll)
        outputs = self.exec_network.infer(inputs=inputs)
        return self.preprocess_output(outputs)

    def check_model(self):
        ### Check for any unsupported layers, and let the user
        ### know if anything is missing. Exit the program, if so.
        supported_layers = self.ie.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            log.error("Unsupported layers found: {}".format(unsupported_layers))
            log.error("Check whether extensions are available to add to IECore.")
            exit(1)

    def preprocess_input(self, left_eye, right_eye, yaw, pitch, roll):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        encoded_left_eye = cv2.resize(left_eye, (self.model_width, self.model_height))
        encoded_left_eye = encoded_left_eye.transpose((2, 0, 1))
        encoded_left_eye = encoded_left_eye.reshape(1, *encoded_left_eye.shape)

        encoded_right_eye = cv2.resize(right_eye, (self.model_width, self.model_height))
        encoded_right_eye = encoded_right_eye.transpose((2, 0, 1))
        encoded_right_eye = encoded_right_eye.reshape(1, *encoded_right_eye.shape)
        return {"left_eye_image": encoded_left_eye, "right_eye_image": encoded_right_eye,
                "head_pose_angles": [yaw, pitch, roll]}

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        return outputs['gaze_vector'][0]
