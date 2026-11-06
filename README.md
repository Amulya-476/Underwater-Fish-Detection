# Underwater Fish Detection 

This project runs your already-trained YOLO fish detector in a Streamlit web app.
The trained model is included at `weights/best.pt`, so you do **not** need to
download the Roboflow dataset or train the model again just to run the app.

## 1. Open the project

1. Open the  `Underwater_Fish_Detection` folder in VS Code.
2. Open **Terminal → New Terminal**.

## 2. Create and activate a virtual environment (Windows)

```powershell
python -m venv venv
venv\Scripts\activate
```

If `python` is not recognized, try `py` instead.

## 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> PyTorch/Ultralytics installation can take a little time. On a CPU-only
> computer, the app can still run, but inference may be slower.

## 4. Run the application

```powershell
python -m streamlit run app.py
```

Streamlit will show a local URL, usually http://localhost:8501.
Open that URL in your browser.

## 5. Use the app

1. Upload a JPG, JPEG, PNG, or WEBP underwater image.
2. Adjust the confidence threshold in the left sidebar if needed.
3. Review the annotated image, total detections, species, and confidence.
4. Download the annotated result if required.

## Project structure

```text
Underwater_Fish_Detection_VSCode/
├── app.py
├── requirements.txt
├── README.md
└── weights/
    └── best.pt
```


