import random
import string
import numpy as np
import cv2
import time

def generate_random_word(length):
    return ''.join(random.choices(string.ascii_uppercase, k=length))

def generate_chart_image(word, index):
    img = np.zeros((2160, 3840, 3), dtype=np.uint8)  # 4K resolution

    font_scale = 18.0 - (index * 1.2)
    font_scale = max(6.0, font_scale)

    text_size = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 30)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = (img.shape[0] + text_size[1]) // 2

    cv2.putText(img, word, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), 30, cv2.LINE_AA)
    return img

CHART_WORDS = [generate_random_word(i + 1) for i in range(10)]
CHART_IMAGES = [generate_chart_image(word, idx) for idx, word in enumerate(CHART_WORDS)]

# Your run_test() function goes below this and assumes CHART_WORDS and CHART_IMAGES are defined
