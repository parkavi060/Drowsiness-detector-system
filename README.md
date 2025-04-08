

```markdown
# Driver Drowsiness Detection System

This project is a real-time Driver Drowsiness Detection System developed using Python, OpenCV, Dlib, and Scipy. It detects when a driver is falling asleep or drowsy by monitoring the Eye Aspect Ratio (EAR) and plays an alarm sound to alert the driver.

## 🧠 How It Works

The system detects the facial landmarks using Dlib’s 68-face-landmark model and continuously calculates the Eye Aspect Ratio (EAR). When the eyes remain closed for a specific number of consecutive frames, the system triggers an alarm.

## 🚗 Features

- Real-time eye monitoring using webcam
- EAR-based blink and drowsiness detection
- Alarm alert system using sound
- Visual feedback with EAR and warning messages

## 🔧 Requirements

- Python 3.x
- OpenCV
- Dlib
- Scipy
- imutils
- playsound

You can install the requirements using:

```bash
pip install opencv-python dlib scipy imutils playsound
```

> ⚠️ Make sure you also have CMake and Visual Studio Build Tools installed (required for Dlib on Windows).

## 📁 Files

- `drowsiness_detector.py`: Main Python script for detection
- `alarm.wav`: Alarm sound file played when drowsiness is detected
- `shape_predictor_68_face_landmarks.dat`: Pre-trained facial landmark model (download link below)

## 📥 Download Shape Predictor Model

You can download the required shape predictor file from [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).

After downloading, extract the `.dat` file and place it in the project directory.

## ▶️ How to Run

```bash
python file_name.py
```

The webcam will open. When your eyes are detected closed beyond a threshold time, an alert message and alarm will be triggered.

Press `ESC` key to exit the application.

## 🖼️ Screenshot

![Demo](demo_picture.png) <!-- Add your own screenshot or video demo -->

## 🧑‍💻 Author

**Parkavi** – [GitHub](https://github.com/parkavi060)  
AI & DS Engineering Student  

