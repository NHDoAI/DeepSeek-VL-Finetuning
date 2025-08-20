## VLLM Navigation for TurtleBot3

End-to-end navigation module that uses a Vision-Language model (DeepSeek-VL 1.3B, QLoRA adapters) to read a front camera image + a small LiDAR text snippet and output one of: straight forward, slow cruise, switch lane. Runs on TurtleBot3 (ROS1 Noetic).

## TL;DR

*   **Input:** RGB image (640×480) + textualized LiDAR (frontal center points).
*   **Output:** high-level decision published by the VLLM to a ROS topic; a small turn node executes it.
*   **Why:** single module replaces hand-engineered perception + rule logic.
*   **Paper:** see citation below for datasets, thresholds, and results.

## Features

*   **Autonomous Decision-Making**: The fine-tuned DeepSeek-VL model makes all high-level driving decisions.
*   **Lane and Obstacle Detection and Avoidance**: Utilizes computer vision to recognize the current lane and the obstacles on the lane.
*   **Distance Categorization**: Classifies the distance to obstacles to inform movement strategy.
*   **LiDAR Data Integration**: Incorporates LiDAR sensor data into the text prompt for more robust environmental perception.
*   **PID Control**: Low-level movement is handled by a PID controller for smooth and precise motion.
*   **ROS Integration**: The entire system is built on the Robot Operating System (ROS), with all components implemented as Python nodes for modularity and communication.

## System Architecture

The system is composed of several interconnected ROS nodes, all written in Python:

1.  **Sensor Nodes**: The standard TurtleBot3 camera and LiDAR nodes publish sensor data to their respective ROS topics.
2.  **DeepSeek-VL Control Node**: 
    *   Subscribes to the camera and LiDAR topics.
    *   Constructs a detailed text prompt by combining the image data with processed LiDAR readings.
    *   Feeds the prompt into the fine-tuned DeepSeek-VL model.
    *   Publishes the model's text-based decision (e.g., "switch lane") to a dedicated command topic.
3.  **PID Controller Node**:
    *   Subscribes to the command topic from the VLLM node.
    *   Receives the text command and maps it to a predefined movement regimen (e.g., executing a lane-switching maneuver).
    *   Publishes velocity commands to the TurtleBot3's motor control topics.

This architecture creates a clear pipeline from perception to action, with the VLLM acting as the central reasoning component.

## Requirements

*   Ubuntu 20.04, ROS1 Noetic
*   Python 3.8+ (training optional: CUDA-enabled PyTorch)
*   Git, CMake, ROS and turtlebot3 build-essential (see Robotis E-manual below)
*   TurtleBot3 Burger with camera + LiDAR (or Gazebo simulation)

## Install

### Clone/fork and install Python requirements

```bash
# clone the repo
git clone https://github.com/cvims/Thesis_Do_Nguyen
```

### Python dependencies (for the VLLM node)

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Make deepseek_vl/ importable
```bash
# install the folder as an editable package
pip install -e .
```
### Install TurtleBot3 dependencies (per e-manual)

Follow the TurtleBot3 Quick Start to install dependent ROS packages and TurtleBot3 stacks for ROS1 Noetic, and to perform robot setup and Wi-Fi configuration:
https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/

### Place the ROS packages in your own catkin workspace

The repo includes the ROS packages (the decision/inference node and the action/turn node). Build them in your catkin workspace:

*   Create your catkin workspace and initialize it
*   Source the Workspace
*   Create the package with the required dependencies:
``` bash
cd ~/catkin_ws/src

catkin_create_pkg cv_bridge geometry_msgs image_transport message_filtersn nav_msgs rospy sensor_msgs std_msgs
catkin_make
source devel/setup.bash
```
*   Add the scripts from the folders ROS_files_for_operation-with-turtlebot3/vllm_controller_pkg/launch/ and ROS_files_for_operation-with-turtlebot3/vllm_controller_pkg/scripts/ to their respective folders in your package. (After adapting the paths to your own setup and environment)
*   Rebuild the workspace.

## Weights & Data

*   **Base:** DeepSeek-VL 1.3B (SigLIP)
*   **Data and Fine-tuned QLoRA adapters:** stored on HPC (computing cluster), in /data/departments/schoen/students/nguyend/MA_Do-Nguyen. Ask Dominik Rößle for more info.
*   Note: The adapter checkpoint that was used for final testing of the system in the paper was the one trained with shortened lidar prompt, batch 6, seed 322, dropout of 0.15 and standard Cross-entropy loss. 


## Quickstart

* Start ros-master with roscore in a terminal on the host machine. Then the following steps:

### Gazebo/simulation version

* Start your gazebo simulation with the turtlebot model loaded and the road world. For example with: roslaunch turtlebot3_gazebo turtlebot3_autorace_2020.launch
* Start the VLLM controller package with: roslaunch vllm_controller_pkg vllm_control.launch real_bot:=false use_compressed:=true
* Note: The performance of the model is directly impacted by how fast the hardware can allow the model to process the inputs. So please note that you will need Hardware fast enough to let the model process the input before the bot collide with the obstacles.

### Real bot version:
* First you need to bringup the bot as described in the E-manual. For example: SSH connect to the bot remotely and then use roslaunch turtlebot3_bringup turtlebot3_robot.launch
* Start the camera module you have installed on the bot SBC. For example in my project I had the raspicam module installed, which could be started with: roslaunch raspicam_node camerav2_1280x960.launch (Adapt to your own setup)
* Then in the host machine: roslaunch vllm_controller_pkg vllm_control.launch real_bot:=true use_compressed:=true


## Training

*   Dataset preparation and exact thresholds are described in the paper and can be found on the HPC in the directory mentioned above.

*   **Scripts:** Different version of the training codes can be found in the folder `training_code/`.

## Repo Layout

*   `deepseek_vl`: contains original source code of the model's API from deepseekVL repo.
*   `evaluation_code`: contains code that were used to evaluate the performance of trained adapters on the eval/test set.
*   `evaluation_results`: contains the results of said evaluations.
*   `inference_code`: contains codes used for inference. (Note: the `inference.py` script in the outer-most level is the original inference code from deepseekVL developer. The codes in the `inference_code` folder were updated to suit this project)
*   `training_code`: contains codes used for Qlora training of base models.
*   `ROS_files_for_operation-with-turtlebot3`: contains codes for ROS nodes that implements the trained model into the movement control nodes. The codes inside need to be placed in a local workspace and built with either catkin or cmake in a system with all ROS requirements setup before they can be used.

## Results

For metrics, comparisons, and ablations, see the paper.

## Citation

```bibtex
@thesis{nguyen2025_vllm_turtlebot3,
  author = {Do H. Nguyen},
  title  = {Intelligent Navigation: Leveraging Vision Large Language Models for Autonomous Movement Control of TurtleBot3},
  school = {Technische Hochschule Ingolstadt},
  year   = {2025},
  month  = {July}
}
```

## License

MIT License.
