#!/usr/bin/env python3
from faceDetection import MPFaceDetection
from faceNet.faceNet import FaceNet
import numpy as np

def detect_face(image: np.array) -> np.array:
    facenet = FaceNet(
        detector = MPFaceDetection(),
        onnx_model_path = "models/faceNet.onnx", 
        force_cpu = True,
    )
    return facenet.detect_save_faces(image)