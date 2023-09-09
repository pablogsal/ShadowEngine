import numpy as np
import matplotlib.pyplot as plt

def read_raw_data(filename, numRows, numCols, numChannels):
    with open(filename, 'rb') as file:
        data = np.fromfile(file, dtype=np.float64, count=numRows * numCols * numChannels)
    return data.reshape((numRows, numCols, numChannels))

if __name__ == "__main__":
    filename = "output.raw"
    numRows = 5000
    numCols = 5000
    numChannels = 3  # Assuming 3 channels for RGB data

    image_data = read_raw_data(filename, numRows, numCols, numChannels)

    # Create an RGB image by stacking channels
    rgb_image = np.zeros((numRows, numCols, 3), dtype=np.uint8)
    for i in range(3):
        rgb_image[:, :, i] = np.clip(image_data[:, :, i] * 255, 0, 255).astype(np.uint8)

    # Display the RGB image
    plt.imshow(rgb_image)
    plt.title('RGB Image')
    plt.axis('off')
    plt.show()
