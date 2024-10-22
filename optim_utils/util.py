import logging
from datetime import datetime
from typing import Tuple
import xml.etree.ElementTree as etree
from xml.dom import minidom

import cv2
import numpy as np
import pydiffvg
import pytz
import torch
from sklearn.cluster import KMeans

def init_diffvg(device: torch.device,
                use_gpu: bool = torch.cuda.is_available(),
                print_timing: bool = False):
    pydiffvg.set_device(device)
    pydiffvg.set_use_gpu(use_gpu)
    pydiffvg.set_print_timing(print_timing)


def setup_logger(output_folder):
    # Create a logger
    logger = logging.getLogger('optimization_log')
    logger.setLevel(logging.INFO)

    # Create a file handler
    local_tz = pytz.timezone('Asia/Shanghai')  # Adjust this to your local timezone
    log_filename = f"{output_folder}/log_{datetime.now(local_tz).strftime('%Y%m%d-%H:%M:%S')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatting for the logs
    class LocalTimeFormatter(logging.Formatter):
        def converter(self, timestamp):
            return datetime.fromtimestamp(timestamp, local_tz)

        def formatTime(self, record, datefmt=None):
            dt = self.converter(record.created)
            if datefmt:
                return dt.strftime(datefmt)
            return dt.strftime("%Y-%m-%d %H:%M:%S")

    formatter = LocalTimeFormatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = etree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def get_most_common_color(image: np.ndarray, contour: np.ndarray) -> Tuple[int, int, int]:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    pixels = masked_image[mask == 255].reshape(-1, 3)
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    most_common_color = colors[counts.argmax()]
    return tuple(map(int, most_common_color))


def get_dominant_color(pixels, n_colors=5):
    # Reshape the pixels to be a list of RGB values
    pixels = pixels.reshape(-1, 3)
    
    n_colors = min(n_colors, len(pixels))
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Get the colors and their counts
    colors = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(kmeans.labels_)
    
    # Return the most common color
    dominant_color = tuple(colors[np.argmax(counts)])
    return dominant_color


def fill_holes(thresh_image):
    thresh_filled = thresh_image.copy()
    h, w = thresh_image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(thresh_filled, mask, (0,0), 255)
    thresh_filled_inv = cv2.bitwise_not(thresh_filled)
    thresh_image = thresh_image | thresh_filled_inv
    return thresh_image
