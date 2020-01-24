import io
import numpy as np
import cv2


def figure_to_numpy(figure):
    buffer = io.BytesIO()
    figure.savefig(buffer, format='png', dpi=90, bbox_inches='tight')
    buffer.seek(0)
    image = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    buffer.close()
    image = cv2.imdecode(image, 1)
    return image
