# Computer Pointer Controller

*TODO:* Write a short introduction to your project

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.
### Install the required Models from OpenVino Model Zoo
1. Download and install de OpenVINO SDK from:
[OpenVINO SFK](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)

2. Setup the environment variable OV with the base path of your OpenVINO installation. Example
        
        export OV=/Users/marcos/intel/openvino

3. Source the OpenVINO environment script:

        source $OV/bin/setupvars.sh

4. Download the four necessary models from OpenVINO Model Zoo

        $OV/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name face-detection-adas-binary-0001
        $OV/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name head-pose-estimation-adas-0001 --precisions FP32
        $OV/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name landmarks-regression-retail-0009 --precisions FP32
        $OV/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name gaze-estimation-adas-0002 --precisions FP32

## Demo
*TODO:* Explain how to run a basic demo of your model.

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
