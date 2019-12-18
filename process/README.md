# ROSBag-Process
Reads, processes, and displays data contained within a ROSBag file

## System Requirements

* Ubuntu 16.04 (native or via Docker)

## Software Requirements

* ROS
* Python 2.7
* XQuartz (if you are running on Mac via Docker)

## Required Python packages

Most of these can be installed with a simple 'pip install MODULE' command

* numpy
* opencv (pip install opencv-python)
* pyglet
* rosbag
* rospkg
* matplotlib (needs version 2.0+)
* cv_bridge (apt-get install ros-kinetic-cv-bridge)

## KITTI to ROSbag conversion

Install Kitti2Bag

'pip install kitti2bag'

To convert, follow the directions on the Github page. Pay attention to the how the zip files are unpacked.

https://github.com/tomas789/kitti2bag

## Command Line

Basic command:

`python extract_rosbag.py <BAGFILE>`

For visualizing only certain topics

`python extract_rosbag.py --topics <TOPIC1>,<TOPIC2>,... <BAGFILE>`

For extracting images

`python extract_rosbag.py --outdir <OUTPUT_DIR> <BAGFILE>`

To run via Docker on Mac:

* Install XQuartz running
* Make sure you have updated XQuartz settings for GLX

`defaults write org.macosforge.xquartz.X11 enable_iglx -bool true`

* Log Out and Log back Into Mac session
* Install socat on your Mac (brew install socat)

Then these steps once per Mac session:

* Run socat on your Mac host

`socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\"`

* Find out your host machine's IP address using 'ifconfig'
* Within your Docker container, setup your DISPLAY environment to point to your host's XQuartz

`export DISPLAY=<HOST_IP_ADDRESS>:0`    
