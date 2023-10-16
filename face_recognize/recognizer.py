from typing import List

import cv2

import numpy as np
import os

import torch
import torchvision


class Recognizer:
    '''
        Recognizer class
    '''
    def __init__(self, path_to_weights=None, device='cpu'):

        # TODO detect faces in image
        self.device = device
        self.face_encoder = torchvision.models.resnet50().to(self.device)

        self.face_detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        self.face_detector.to(device)
        self._load_weights(path_to_weights)

    def _load_weights(self, path):
        face_encoder_weights = torch.load(os.path.join(path, 'face_encoder.pt'))
        face_detector_weights = torch.load(os.path.join(path, 'fasterrcnn_resnet50_fpn.pth'))

        self.face_encoder.load_state_dict(face_encoder_weights)
        self.face_detector.load_state_dict(face_detector_weights)

    def recognize(self, image: np.ndarray):
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        image_tensor = torch.from_numpy(image).to(self.device)

        embedding = self.face_encoder(image_tensor)
        embedding = embedding.detach().numpy()

        return embedding

    def detect(self, image: np.ndarray, thresh=0.3) -> List:
        """Return cropped face image."""
        image_tensor = torch.from_numpy(image).to(self.device)

        with torch.no_grad():
            predictions = self.face_detector(image_tensor.unsqueeze(0))[0].cpu()

        boxes = predictions['boxes']
        labels = predictions['labels']
        scores = predictions['scores']

        return [box for box, score in zip(boxes, scores) if scores > thresh]
