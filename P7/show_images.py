from matplotlib import pyplot as plt

def show_images(images, titles):
    for i in range(len(images)):
        plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray',)
        plt.title(titles[i])
        plt.axis(False)
    plt.show()