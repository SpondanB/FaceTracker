# 🧑‍🎨 Face Avatar Generator using Facial Landmarks

A real-time **face avatar generation system** built using **MediaPipe Face Mesh**, **OpenCV**, and **Pygame**. This project maps live facial landmarks captured from a webcam to a stylized 2D avatar rendered programmatically, demonstrating how computer vision outputs can drive real-time graphical representations.

This repository represents an **early but complete prototype** of a virtual avatar pipeline and serves as a foundation for more advanced avatar and facial animation systems.

---

## 📌 Project Overview
The goal of this project is to:
- Detect **facial landmarks** in real time using MediaPipe
- Extract meaningful facial regions (eyes, nose, mouth, eyebrows, face outline)
- Render a dynamic **2D avatar** using geometric primitives in Pygame

Instead of drawing raw landmark points, the system groups landmarks into semantic facial components and renders them as filled polygons, producing a coherent avatar that mirrors the user’s facial movements.

---

## 🧠 Core Concepts
- Facial landmark detection (MediaPipe Face Mesh)
- Real-time webcam processing with OpenCV
- Landmark grouping and geometric interpretation
- Procedural avatar rendering using polygons
- Synchronization between CV pipelines and rendering loops

---

## 🎨 Facial Components Rendered
The avatar is composed of multiple facial regions derived from MediaPipe landmarks:

- **Face outline & cheeks** – base facial structure
- **Eyes & eyelids** – eye whites and blinking regions
- **Irises** – eye tracking approximation
- **Eyebrows** – expressive upper-face features
- **Nose (upper & lower)** – depth cueing via shading
- **Mouth** – dynamically positioned based on lip landmarks

Each component is rendered using carefully selected landmark indices to preserve facial proportions.

---

## 🧩 Implementation Details

### Facial Landmark Processing
- MediaPipe Face Mesh detects **468+ refined landmarks**
- Selected landmark indices are grouped into facial regions
- Normalized landmark coordinates are mapped to screen space

### Avatar Rendering
- Pygame is used to draw filled polygons for each facial feature
- Layered drawing ensures correct visual stacking (face → eyes → nose → mouth)
- Color palettes simulate basic skin tones and contrast

### Real-Time Loop
- Webcam frame capture (OpenCV)
- Face landmark inference (MediaPipe)
- Polygon-based avatar rendering (Pygame)

---

## 🛠️ Requirements
- Python 3.8+
- Webcam
- Required libraries:
  - pygame
  - opencv-python
  - mediapipe

Install dependencies:
```bash
pip install pygame opencv-python mediapipe
```

---

## ▶️ Running the Prototype
```bash
python FaceAvatar.py
```

Ensure your face is clearly visible to the webcam for accurate landmark tracking.

---

## 📈 Skills Demonstrated
- Computer vision with MediaPipe
- Facial landmark analysis
- Real-time graphics rendering
- Geometric modeling of facial features
- Multi-library integration in Python

---

## 🧠 Learning Outcomes
- Understanding dense facial landmark representations
- Translating CV outputs into graphical avatars
- Managing performance constraints in real-time systems
- Designing modular facial feature pipelines

---

## 🏷️ Portfolio Context
This project represents an **early-stage implementation** of a broader idea: building a robust virtual avatar system driven by facial tracking.

The concept was later expanded and refined in a more advanced project:

➡️ **Virtual Skull Avatar (Improved Version)**  
🔗 https://github.com/SpondanB/Virtual-Skull-Avatar

The final outcome of this repository is preserved in the **"Old Version"** directory of the newer project for reference and comparison.

---

## 🔮 Future Improvements
- Facial expression classification
- Smooth interpolation and animation
- 3D avatar extensions
- Emotion-aware rendering
- Performance optimization

---

**Author:** Spondan Bandyopadhyay
**Interests:** Computer Vision, Graphics, AI Systems

---
