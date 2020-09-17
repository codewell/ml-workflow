from itertools import product
from PIL import ImageFont


def text_(draw, text, x, y, fill='black', outline='white', size=12):
    font = ImageFont.load_default()

    for x_shift, y_shift in product([-1, 0, 1], [-1, 0, 1]):
        draw.text((x + x_shift, y + y_shift), text, font=font, fill=outline)

    draw.text((x, y), text, font=font, fill=fill)
