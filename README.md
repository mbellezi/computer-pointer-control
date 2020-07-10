# Computer Pointer Controller

*TODO:* Write a short introduction to your project

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
        
5. Install the environment and libraries

        source env/bin/activate
        pip install -r requirements.txt

## Demo
After install the models, cd to the project main directory.
You can run the demo with the commands:
        
        source env/bin/activate
        python3 src/main.py -t video -i bin/demo.mp4 -p

## Documentation
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

```

You can use the `--type` cam to use your camera feed instead of the video file.

The `--preview` option can be used to view the extracts od the face and eyes.

The `--notmove` can be used to disable the mouse movement to get a better benchmark.

The `--model` controls the precision of the models used. One of FP32, FP16 or FP16-INT8


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

The total inference time for the FP32 is 11.44 ms with 87.41 fps. This is a very
interesting performance with a near real time feeling.
