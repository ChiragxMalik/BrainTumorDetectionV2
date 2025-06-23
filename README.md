# Brain Tumor Detection Using Deep Learning

A web application for detecting brain tumors from MRI images using a deep learning model.  
The app features a modern, responsive Material UI-inspired frontend and a simple Flask backend for predictions.

## Features

- Upload MRI images and get instant predictions (tumor detected or not).
- Clean, modern Material UI-inspired design with drag-and-drop upload, image preview, and theme toggle (light/dark).
- Powered by a pre-trained Keras/TensorFlow model.

## Demo

![App Screenshot](screenshot.png) <!-- Add a screenshot if you have one -->

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/chiragxmalik/brain-tumor-detection.git
cd brain-tumor-detection
```

### 2. Install dependencies

It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the app

```bash
python app.py
```

Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## Project Structure

```
.
├── app.py
├── BrainTumorDetectionModel.h5
├── requirements.txt
├── templates/
├── static/
├── uploads/
├── datasets/
├── pred/
├── mainTrain.py
├── mainTest.py
└── README.md
```

## Model

- The model (`BrainTumorDetectionModel.h5`) is a Keras/TensorFlow model trained to classify MRI images as having a brain tumor or not.
- If you want to retrain or improve the model, see `mainTrain.py` and `mainTest.py`.

## Credits & AI Assistance

This project was originally created as a classic deep learning web app.  
Recently, I used AI prompts (with GPT-4) to make my old project more efficient and to give it a much better, modern frontend.  
It's amazing how just a couple of prompts can transform and modernize a project!

## License

[MIT](LICENSE) 