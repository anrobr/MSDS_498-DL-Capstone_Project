## MSDS 498-DL Capstone Project
### Advanced Driver Assistance System using Intelligent Video Analytics on the Edge

![Project image](https://github.com/anrobr/MSDS_498-DL-Capstone_Project/blob/master/header_image.png?raw=true)

--- 
### Background
According to the WHO 
* around 1.3 mio people die and
* around 50 mio people are injured
every year from traffic accidents.

Around **93%** of the world's fatalities are happening in **low- and middle-income countries**.

---
### Project Goal
This project aims to reduce the number and severity of traffic accidents caused by distracted driving, one of the main causes of traffic accidents, and to contribute to the WHO’s ambitious goal to cut traffic accidents in half by 2030 as filed via resolution A/RES/74/299.

---
### Approach
We aim to provide an Advanced Driver Assistance System (ADAS) based on Intelligent Video Analytics (IVA) that can be deployed as a self-sufficient, **affordable**, and model-agnostic unit (edge system) in motor vehicles such as cars, vans, and mini trucks. The system is designed for deployment using Raspberry Pi 4 and the Intel Neural Compute Stick 2 (NCS2). The model inference is done by the NCS2. To be useful for the described purpose the throughput of the system is >= 30 FPS. According to our estimates, the system should be affordable enough for customers in low- and middle-income countries.

---
### Tags
[Computer vision, object detection, object tracking, deep-learning, real-time, edge AI, OpenVINO, OpenVINO Model Zoo, Intel Neural Compute Stick 2, NCS2.]

---
### Repository Contents

* documentation
  * Paper.pdf
  * Presentation.pdf

* implementation
  * proof-of-concept (_folder containing files to evaluate required neural networks and their dependencies_)
    * **proof-of-concept.ipynb** (_Jupyter notebook used for the evaluation of models to determine the gaze of a driver_)
  * model-building (_folder containing files to build custom models_)
    * closed-eyes-detection (_folder containing files to build a custom closed eye detector based on TensorFlow 2.x_)
        * **model_building.ipynb** (_Jupyter notebook to build a TensorFlow 2.x model and export it for use with OpenVINO and the NCS2_)
  * distracted-driving-detection (_folder containing the implementation of the Advanced Driver Assistance System (ADAS)_)
    * **main.py** (_entry point of the ADAS_)
    * models (_folder used for storing and converting models_)
    * **models_setup.ipynb** (_Jupyter notebook to download pretrained models from Open Model Zoo or convert custom ones to OpenVINO Intermediate Representation (IR)_)
    * output (_folder containing video recordings with the annotated estimated gaze of a driver_) 
      * NCS2_MYRIAD (_folder containing annotated recordings to demonstrate the ADAS in action using recorded video data of a driver_)
