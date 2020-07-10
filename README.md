# Computer Pointer Controller

This project is a small prototype for testing the detection of the gaze direction of one person and use it to control
the direction of the user's mouse pointer. This project uses the OpenVINO SDK and several models from the openVINO zoo
to create a pipeline of inference. The first module used is the Face detection, that is used to extract the face image
from the video feed. This face image as input to the Facial Landmark model that outputs the face landmark features as 
the eyes positions. The eye's positions are use to extract both eyes from the image. The Head pose model is used to
to detect the pose of the face's head. Later, both the eyes images and the pose information (yaw, tilt and roll angles)
are used as inputs to the Gaze estimation model and the it's results (gaze) are used to control the mouse pointer
scrolling direction.

![Diagram](https://video.udacity-data.com/topher/2020/April/5e923081_pipeline/pipeline.png)
(image by Udacity/Intel)

## Project Set Up and Installation
### Install the required Models from OpenVino Model Zoo
1. Download and install de OpenVINO SDK from:
[OpenVINO SFK](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)

2. Setup the environment variable OV with the base path of your OpenVINO installation. Example
        
        export OV=/Users/marcos/intel/openvino

3. Source the OpenVINO environment script:

        source $OV/bin/setupvars.sh

4. Download the four necessary models from OpenVINO Model Zoo

        $OV/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name face-detection-adas-binary-0001 -o models
        $OV/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name head-pose-estimation-adas-0001 -o models
        $OV/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name landmarks-regression-retail-0009 -o models
        $OV/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name gaze-estimation-adas-0002 -o models
        
5. Install VirtualEnv and  and libraries
        pip install virtualenv
        virtualenv env
        source env/bin/activate
        pip install -r requirements.txt

## Demo
After install the models, cd to the project main directory.
You can run the demo with the commands:
        
        source env/bin/activate
        python3 src/main.py -t video -i bin/demo.mp4 -p

## Documentation
The directory structure is the following:
```
|- README.md                       - This file
|- bin
|---- demo.mp4                     - Sample video for testing the inference
|- env
|---- bin                          - virtual environment setup
|- models
|---- intel                        - OpenVINO Zoo used Models
|- requirements.txt                - Python library required
|- src
|- facial_landmark_detection.py    - implementation of the facial landmark detection
|- head_pose_estimation.py         - implementation of the pose estimation
|- main.py                         - main application
|- face_detection.py               - implementation of the face detection model
|- gaze_estimation.py              - implementation of the gaze estimation model
|- input_feeder.py                 - module for loading the video or camera feed
|- mouse_controller.py             - module for controlling the mouse
```

The demo can use this command line options:
```
usage: main.py [-h] -t INPUT_TYPE -i INPUT [-o OUT] [-p] [--notmove]

optional arguments:
  -h, --help            show this help message and exit
  -t INPUT_TYPE, --input-type INPUT_TYPE
                        Type of input (video or cam)
  -i INPUT, --input INPUT
                        Input file
  -o OUT, --out OUT     Output file with the processed content
  -p, --preview         Should preview face and eyes
  --notmove             Should not move mouse
  -m MODEL, --model MODEL
                        Model precision to use. One of FP32, FP16 or FP16-INT8
  -d DEVICE, --device DEVICE
                        Device used to process model. One or CPU or GPU
  -v, --verbose         Enable DEBUG messages
```

You can use the `--type` cam to use your camera feed instead of the video file.

The `--preview` option can be used to view the extracts od the face and eyes.

The `--notmove` can be used to disable the mouse movement to get a better benchmark.

The `--model` controls the precision of the models used. One of FP32, FP16 or FP16-INT8

The `--device` controls the Intel hardware used for loading the models. Could be CPU, GPU or VPU if they are available. 

The `--verbose` controls the displaying of verbose debug messages. 


## Benchmarks

This tests were made on a MacBook Pro with a i9 processor. As the OpenVINO on MacOS does not allow use of the GPU,
 the tests were made only using the CPU. The models were test for the FP32, FP16 and FP16-INT8 precisions, except
 the Face detection tha was only available using FP32-INT1.

|  Model      | Load Time (ms) | Average inference time (ms) |
| ----------- | ----------- | ----------- |
| Face detection - FP32     | 5134.97       | 8.63ms       |
| Facial landmark detection - FP32  | 302.16        | 0.51ms       |
| Head Pose estimation - FP32  | 617.12        |  1.04ms       |
| Gaze estimation - FP32  |  748.72        |  1.26ms       |
| Facial landmark detection - FP16  | 296.52        |  0.50ms       |
| Head Pose estimation - FP16  | 620.61        |  1.04ms       |
| Gaze estimation - FP16  |  750.20        |  1.26ms       |
| Facial landmark detection - FP16-INT8  | 297.25        |  0.50ms       |
| Head Pose estimation - FP16-INT8  | 497.46        |   0.84ms       |
| Gaze estimation - FP16-INT8  |  601.19        |  1.01ms       |

## Results

As can be seen from the results, there is a slight difference between the FP32 and FP16 on CPU. 
The major differences are seen when using the FP16-INT8. The models Head pose estimation and Gaze estimation
have the greater increase in performance when using the FP16-INT8.

All the models with FP16 and FP16-INT8 may see a reduction on loading times because they are quantized versions with 
weights occupying half or a quarter of memory space than the FP32 weights.  

The FP16 and FP16-INT8 may have smaller inference times because they use smaller data types that takes less space and 
in some cases can runs more than one calculation on onees instruction.  

The total inference time for the FP32 is 11.44 ms with 87.41 fps. This is a very
interesting performance with a near real time feeling.
