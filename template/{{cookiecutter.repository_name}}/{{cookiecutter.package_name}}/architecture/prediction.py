from itertools import product
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel
from typing import Tuple

from {{cookiecutter.package_name}} import problem


def text_(draw, text, x, y, fill='black', outline='white', size=12):
    font = ImageFont.load_default()

    for x_shift, y_shift in product([-1, 0, 1], [-1, 0, 1]):
        draw.text((x + x_shift, y + y_shift), text, font=font, fill=outline)

    draw.text((x, y), text, font=font, fill=fill)


def class_index(class_name):
    return problem.settings.CLASS_NAMES.index(class_name)


class Prediction(BaseModel):
    logits: torch.Tensor
    preprocessed: torch.Tensor

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    def class_name(self):
        return problem.settings.CLASS_NAMES[self.logits.argmax()]

    def image(self):
        return Image.fromarray(np.uint8(
            (self.preprocessed.squeeze(0).cpu().numpy() + 1) / 2 * 255
        ))

    def representation(self):
        image = self.image().copy().resize((256, 256))

        probabilities = dict(zip(
            problem.settings.CLASS_NAMES,
            self.logits.sigmoid().detach().cpu().numpy(),
        ))

        draw = ImageDraw.Draw(image)
        for index, (class_name, probability) in enumerate(probabilities.items()):
            text_(draw, f'{class_name}: {probability:.2f}', 10, 5 + 10 * index)
        return image

    @property
    def _repr_png_(self):
        return self.representation()._repr_png_


class PredictionBatch(BaseModel):
    logits: torch.Tensor
    preprocessed: torch.Tensor

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    def __len__(self):
        return len(self.logits)

    def __getitem__(self, index):
        return Prediction(
            logits=self.logits[index],
            preprocessed=self.preprocessed[index],
        )

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def loss(self, class_names):
        targets = torch.tensor(
            [class_index(class_name) for class_name in class_names],
        ).to(self.logits).long()
        return F.cross_entropy(self.logits, targets)

    def cpu(self):
        return PredictionBatch(
            logits=self.logits.cpu(),
            preprocessed=self.preprocessed.cpu(),
        )

    def detach(self):
        return PredictionBatch(
            logits=self.logits.detach(),
            preprocessed=self.preprocessed.detach(),
        )
