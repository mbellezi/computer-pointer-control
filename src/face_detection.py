import cv2
import logging as log
from openvino.inference_engine import IENetwork, IECore

'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''


class Model_Face_Detection:
    '''
    Class for the Face Detection Model.
    '''

    def __init__(self, model_name, conf=0.5, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.confidence = conf
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
        outputs = []
        frame_inference = self.preprocess_input(image)

        # Start asynchronous inference for specified request
        self.exec_network.start_async(request_id=0, inputs={self.input_name: frame_inference})
        if self.exec_network.requests[0].wait(-1) == 0:
            outputs = self.preprocess_output(self.exec_network.requests[0].outputs[self.output_name])
        return outputs

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

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        # Return only the first face
        if outputs[0][0][0] is not None and outputs[0][0][0][2] >= self.confidence:
            return outputs[0][0][0]
        else:
            return None
