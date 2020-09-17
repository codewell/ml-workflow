from itertools import product
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel
from typing import Tuple

from {{cookiecutter.package_name}} import problem, tools


class Prediction(BaseModel):
    logits: torch.Tensor

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    @property
    def probabilities(self):
        return self.logits.detach().cpu().sigmoid()

    @property
    def class_name(self):
        return problem.settings.CLASS_NAMES[self.logits.argmax()]

    def representation(self, example=None):
        if example:
            image = example.image.copy().resize((256, 256))
        else:
            image = Image.new('L', (256, 256))

        probabilities = dict(zip(
            problem.settings.CLASS_NAMES,
            self.probabilities,
        ))

        draw = ImageDraw.Draw(image)
        for index, (class_name, probability) in enumerate(probabilities.items()):
            tools.text_(
                draw,
                f'{class_name}: {probability:.2f}',
                10,
                5 + 10 * index
            )
        return image

    @property
    def _repr_png_(self):
        return self.representation()._repr_png_


class PredictionBatch(BaseModel):
    logits: torch.Tensor

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    def __len__(self):
        return len(self.logits)

    def __getitem__(self, index):
        return Prediction(
            logits=self.logits[index],
        )

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]
    
    @property
    def probabilities(self):
        return self.logits.detach().cpu().sigmoid()

    def stack_class_indices(self, examples):
        return torch.as_tensor(
            np.stack([example.class_index for example in examples]),
            device=self.logits.device
        )

    def loss(self, examples):
        return self.cross_entropy(examples)
    
    def cross_entropy(self, examples):
        return F.cross_entropy(
            self.logits,
            self.stack_class_indices(examples),
        )

    def cpu(self):
        return PredictionBatch(**{
            name: (
                value.cpu() if isinstance(value, torch.Tensor)
                else [v.cpu() for v in value] if type(value) == list
                else value
            )
            for name, value in super().__iter__()
        })

    def detach(self):
        return PredictionBatch(**{
            name: (
                value.detach() if isinstance(value, torch.Tensor)
                else [v.detach() for v in value] if type(value) == list
                else value
            )
            for name, value in super().__iter__()
        })
