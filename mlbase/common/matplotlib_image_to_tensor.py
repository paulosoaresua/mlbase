import matplotlib.pyplot as plt
import io
import PIL
from torchvision.transforms import ToTensor


def matplotlib_image_to_tensor(figure: plt.figure):
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(figure)
    buffer.seek(0)

    image = PIL.Image.open(buffer)

    ToTensor()(image)
