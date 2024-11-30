# Welcome to VisionForge
[![pylint]()](https://redirect/link)

Project Requires:
- Python 3.10
- Cuda 11.8
- Cudnn >= 8.7
- GPU VRAM >= 24GB
- opencv-headless
- torch 2.0
- ultralytics/yolov5
- accelerate
- insightface
- onnxruntime-gpu
- numpy == 1.20 (must be this version to support insightface)
- pip install tensorflow[and-cuda]

## Installation steps
1. Clone the repository to your local machine 

````
git clone https://github.com/forms-dev/VisionForge.git
````

2. Change permissions on the VisionForge directory

````
sudo chmod u+x VisionForge
sudo chmod u+x VisionForge/setup.sh
````

3. Run the setup script

````
cd VisionForge
./setup.sh
````

Once the setup script has completed, you will have a fully functional VisionForge environment. The setup scripts runs a 
local host server to build documents. Keep an eye on the terminal for the local host address.
#   V i s i o n F o r g e  
 