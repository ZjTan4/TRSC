import numpy as np

def rgb2mask(img, color_id):
    # assert len(img) == 3
    h, w, c = img.shape
    out = np.ones((h, w, c)) * 255
    for i in range(h):
        for j in range(w):
            if tuple(img[i, j]) in color_id:
                out[i][j] = color_id[tuple(img[i, j])]
            else:
                print("unknown color, exiting...")
                exit(0)
    return out