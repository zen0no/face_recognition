from typing import List

import cv2

import numpy as np
import os

import torch
import torchvision


class Recognizer:
    def __init__(self, path_to_weights=None, device='cpu'):

        # TODO detect faces in image
        self.device = device
        self.face_encoder = torchvision.models.resnet50().to(self.device)

        self.face_detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        self._load_weights(self.face_detector, "fasterrcnn_resnet50_fpn.pth")
        self.face_detector.to(device)

    @staticmethod
    def _load_weights(model, path):
        face_encoder_weights = torch.load(os.path.join(path,
                                                       'face_encoder.pt'))
        model.load_state_dict(face_encoder_weights)

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
