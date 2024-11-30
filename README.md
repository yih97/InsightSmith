# Welcome to VisionForge  

---

### **Project Requirements**  
- **Python**: 3.10  
- **CUDA**: 11.8  
- **cuDNN**: >= 8.7  
- **GPU VRAM**: >= 24GB  
- **Libraries**:  
  - `opencv-headless`  
  - `torch` 2.0  
  - `ultralytics/yolov5`  
  - `accelerate`  
  - `insightface`  
  - `onnxruntime-gpu`  
  - `numpy` == 1.20 *(mandatory for `insightface` support)*  
  - `tensorflow[and-cuda]` *(install using pip)*  

---

## **Installation Steps**  

### 1. **Clone the Repository**  
Run the following command to clone the VisionForge repository to your local machine:  
```bash
git clone https://github.com/forms-dev/VisionForge.git
```

---

### 2. **Change Permissions**  
Update permissions for the VisionForge directory and setup script:  
```bash
sudo chmod u+x VisionForge
sudo chmod u+x VisionForge/setup.sh
```

---

### 3. **Run the Setup Script**  
Navigate to the VisionForge directory and execute the setup script:  
```bash
cd VisionForge
./setup.sh
```

---

### **Post-Setup Instructions**  
Once the setup is complete:  
- A **local host server** will be started to build documents.  
- Keep an eye on the **terminal output** for the local host address.  

Your VisionForge environment is now ready to use!  

---  
