import numpy as np
import rasterio

def split_image(image, chunk_size):
    """
    Split the given image into smaller chunks.

    Parameters:
    image (numpy.ndarray): The input image with shape (height, width, channels).
    chunk_size (tuple): The size of each chunk in format (height, width).

    Returns:
    List of numpy.ndarray: List of image chunks.
    """
    chunks = []
    height, width, channels = image.shape
    chunk_height, chunk_width = chunk_size

    # Calculate the number of chunks needed in each dimension
    num_chunks_height = height // chunk_height
    num_chunks_width = width // chunk_width

    # Iterate through each chunk
    for i in range(num_chunks_height):
        for j in range(num_chunks_width):
            # Extract the chunk from the image
            chunk = image[i * chunk_height:(i + 1) * chunk_height, j * chunk_width:(j + 1) * chunk_width, :]
            chunks.append(chunk)

    return chunks


def merge_image(image_chunks, original_shape):
    """
    Merge the given image chunks into the original image.

    Parameters:
    image_chunks (list): List of image chunks.
    original_shape (tuple): The original shape of the image in format (height, width, channels).

    Returns:
    numpy.ndarray: The merged image.
    """
    height, width, channels = original_shape
    merged_image = np.zeros(original_shape, dtype=image_chunks[0].dtype)

    # Initialize indices for placing chunks
    i_chunk = 0
    for i in range(height // image_chunks[0].shape[0]):
        for j in range(width // image_chunks[0].shape[1]):
            merged_image[i * image_chunks[0].shape[0]:(i + 1) * image_chunks[0].shape[0],
                         j * image_chunks[0].shape[1]:(j + 1) * image_chunks[0].shape[1], :] = image_chunks[i_chunk]
            i_chunk += 1

    return merged_image



if __name__ == "__main__":

    with rasterio.open("d:\\Ramesh\\20230401120000_3.25m_DN_Jind_Denali_MS\\20230401120000_3.25m_DN_Jind_Denali_MS.tif") as src :
        image = src.read()
        original_shape = image.shape
    chunk_size = (1024, 1024)
    image_chunks = split_image(image, chunk_size)

    merged_image = merge_image(image_chunks, original_shape)
    print(len(image_chunks))
    print(merged_image.shape)