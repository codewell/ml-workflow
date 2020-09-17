from itertools import product
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pydantic import BaseModel

from {{cookiecutter.package_name}} import problem, tools


class Example(BaseModel):
    image: Image.Image
    class_name: str

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    @property
    def class_index(self):
        return problem.settings.CLASS_NAMES.index(self.class_name)

    def representation(self):
        image = self.image.copy()
        draw = ImageDraw.Draw(image)
        tools.text_(draw, self.class_name, 10, 10)
        return image

    @property
    def _repr_png_(self):
        return self.representation()._repr_png_

    def augmented(self, augmenter):
        image = Image.fromarray(
            augmenter.augment(image=np.array(self.image))
        )
        return Example(
            image=image,
            class_name=self.class_name
        )
