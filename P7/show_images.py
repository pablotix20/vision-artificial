from matplotlib import pyplot as plt
import math


def show_images(images, titles):
    n = math.ceil(math.sqrt(len(images)))
    for i in range(len(images)):
        plt.subplot(n, n, i+1), plt.imshow(images[i], 'gray',)
        plt.title(titles[i])
        plt.axis(False)
    plt.show()
