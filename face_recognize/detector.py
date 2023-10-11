import cv2

import numpy as np
import os

import torch
import torchvision


class Detector:
    def __init__(self, path_to_weights=None, device='cpu'):

        # TODO detect faces in image
        self.device = device
        self.face_encoder = torchvision.models.resnet50().to(self.device)

    def _load_weights(self, path):
        face_encoder_weights = torch.load(os.path.join(path,
                                                       'face_encoder.pt'))
        self.face_encoder.load_state_dict(face_encoder_weights)

    def detect(self, image: np.ndarray):
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        image_tensor = torch.from_numpy(image).to(self.device)

        embedding = self.face_encoder(image_tensor)
        embedding = embedding.detach().numpy()

        return embedding
