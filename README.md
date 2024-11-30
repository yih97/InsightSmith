# Welcome to VisionForge  

---

### **프로젝트 개요**  
**VisionForge**는 **Diffusion Model**을 기반으로, 사용자가 사진을 업로드하면 다양한 테마에 맞춘 나만의 AI 사진을 생성해주는 서비스입니다.  
이 서비스를 통해 창의적이고 맞춤형 이미지 생성 경험을 제공합니다.  
**단, 상업적 목적으로의 사용은 금지됩니다.**

---

### **Project Requirements**  
- **Python**: 3.10  
- **CUDA**: 11.8  
- **cuDNN**: >= 8.7  
- **GPU VRAM**: >= 24GB  
- **필수 라이브러리**:  
  - `opencv-headless`  
  - `torch` 2.0  
  - `ultralytics/yolov5`  
  - `accelerate`  
  - `insightface`  
  - `onnxruntime-gpu`  
  - `numpy` == 1.20 *(insightface 지원 필수 버전)*  
  - `tensorflow[and-cuda]` *(pip으로 설치)*  

---

## **Installation Steps**  

### 1. **Clone the Repository**  
아래 명령어를 실행하여 VisionForge 레포지토리를 로컬에 복제합니다:  
```bash
git clone https://github.com/forms-dev/VisionForge.git
```

---

### 2. **Change Permissions**  
VisionForge 디렉터리와 설치 스크립트의 실행 권한을 변경합니다:  
```bash
sudo chmod u+x VisionForge
sudo chmod u+x VisionForge/setup.sh
```

---

### 3. **Run the Setup Script**  
VisionForge 디렉터리로 이동하여 설치 스크립트를 실행합니다:  
```bash
cd VisionForge
./setup.sh
```

---

### **Post-Setup Instructions**  
- 설치가 완료되면 **로컬 호스트 서버**가 시작되어 문서 생성 환경이 구성됩니다.  
- **터미널 출력**을 확인하여 로컬 호스트 주소를 확인하세요.  

---

### **VisionForge의 특징**  
- **Diffusion Model**을 사용하여 고품질 AI 이미지를 생성.  
- 사용자가 업로드한 사진을 분석해 **테마 기반 맞춤형 이미지**를 제공합니다.  
- 다양한 테마와 스타일을 통해 창의적인 이미지 제작 가능.  

---

### **중요 안내**  
- VisionForge는 **비상업적 목적**으로만 사용 가능합니다.  
- 상업적 목적으로의 사용은 **엄격히 금지**됩니다.  

지금 VisionForge를 설치하고 나만의 AI 사진을 만나보세요!  
