# 🚦 Traffic Analytics System using YOLO

An end-to-end **Computer Vision** application that performs **vehicle detection**, **vehicle counting**, and **traffic density estimation** from road traffic images using **YOLOv2**, TensorFlow, and Streamlit.

The application automatically downloads the trained YOLO model from Hugging Face during the first execution and provides real-time traffic analytics through an interactive web interface.

---

## 📌 Features

- 🚗 Multi-class vehicle detection using YOLOv2
- 🚙 Vehicle counting
- 📊 Class-wise vehicle statistics
- 🚦 Traffic density estimation (Low / Medium / High)
- 🖼️ Bounding-box visualization
- 🌐 Interactive Streamlit web application
- 🤗 Automatic model download from Hugging Face

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- YOLOv2
- OpenCV
- NumPy
- Pillow
- Streamlit
- Hugging Face Hub

---

# 🏗️ System Architecture

```text
               Upload Image
                     │
                     ▼
           Streamlit Web Interface
                     │
                     ▼
          Download Model (Hugging Face)
                     │
                     ▼
           TensorFlow YOLOv2 Model
                     │
                     ▼
             Object Detection
                     │
                     ▼
        Vehicle Classification & Counting
                     │
                     ▼
        Traffic Density Estimation
                     │
                     ▼
        Detection Result + Analytics
```

---

# 📂 Project Structure

```text
Traffic-Analytics-System-Using-YOLO/
│
├── app.py
├── Traffic_Analytics_System_using_YOLO.py
├── download_model.py
├── requirements.txt
├── README.md
│
├── font/
├── images/
├── model_data/
├── yad2k/
│
└── outputs/
```

---

# 📸 Screenshots

### Original Image

```
out/modern_city.webp

```

---

### Detection Result


```
out/modern_city-op.webp
```


# 🚀 Live Demo

🌐 **Streamlit App**

> https://your-streamlit-app.streamlit.app

---

# 🤗 Model

The trained YOLOv2 model is hosted on Hugging Face.

Model Repository

https://huggingface.co/SyedHarshath/traffic-analytics-yolo

---

# 📈 Traffic Analytics

The application provides:

- Vehicle Detection
- Vehicle Counting
- Class-wise Statistics
- Traffic Density Estimation
- Detection Visualization

Supported vehicle classes:

- Car
- Bus
- Truck
- Motorbike

Traffic Density Levels

- 🟢 Low
- 🟡 Medium
- 🔴 High

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/SyedHarshath/Traffic-Analytics-System-Using-YOLO.git

cd Traffic-Analytics-System-Using-YOLO
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

The model will be downloaded automatically from Hugging Face during the first run.

---

# 📚 Learning Resources

This project was developed by applying concepts learned from the **Convolutional Neural Networks** course offered by **DeepLearning.AI**, with additional implementation and deployment work completed independently.

---

# 👨‍💻 Author

**Syed Harshath**

LinkedIn:
https://www.linkedin.com/in/syed-harshath-46244128a/

GitHub:
https://github.com/SyedHarshath

---

# ⭐ If you found this project useful

Please consider giving this repository a ⭐ on GitHub.