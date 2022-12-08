import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch

def main():
    # test
    image_name = "creek_00086.png"
    image = Image.open("/home/ztan4/Documents/TRSC/data/RUGD/images/{}".format(image_name))
    image = np.array(image)
    gray_image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    glcm_np = get_glcm_np(gray_image)
    # glcm = get_glcm(torch.tensor(gray_image))
    # a = torch.Tensor(glcm_np - glcm.numpy()).unique()
    plt.subplot(1, 2, 1)
    plt.imshow(gray_image / 255)
    glcm_mean = get_glcm_mean_np(glcm_np)
    plt.subplot(1, 2, 2)
    plt.imshow(glcm_mean / 255)
    plt.show()
    # for i in range(3):
    #     im = image[:, :, i]
    #     glcm_mean = get_glcm_mean_np(im)
    #     plt.imshow(im)
    #     plt.imshow(glcm_mean)
    #     plt.show()

def get_glcm(img, vmin=0, vmax=255, nbit=8, kernel_size=5):
    mi, ma = vmin, vmax
    ks = kernel_size
    h,w = img.shape

    # digitize
    bins = torch.linspace(mi, ma+1, nbit+1)
    # gl1 = np.digitize(img, bins) - 1
    gl1 = torch.bucketize(img, bins) - 1 # digitize
    gl2 = torch.cat((gl1[:,1:], gl1[:,-1:]), dim=1) # append

    # make glcm
    glcm = np.zeros((nbit, nbit, h, w), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            mask = ((gl1==i) & (gl2==j))
            glcm[i,j, mask] = 1

    kernel = np.ones((ks, ks), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            glcm[i,j] = cv2.filter2D(glcm[i,j], -1, kernel)

    glcm = torch.Tensor(glcm).float()
    return glcm

def get_glcm_features(glcm, kernal_size=5):
    '''
    0 - mean
    1 - std
    2 - contrast
    3 - dissimilarity
    4 - homogeneity
    5 - asm
    6 - energy
    7 - max
    8 - entropy
    '''
    nbit, _, h, w = glcm.shape
    num_features = 9
    features = torch.zeros((num_features, h, w), dtype=torch.float32)
    for i in range(nbit):
        for j in range(nbit):
            pass


# Adapted from reference: https://github.com/1044197988/Python-Image-feature-extraction/blob/master/%E7%BA%B9%E7%90%86%E7%89%B9%E5%BE%81/GLCM/fast_glcm.py
def get_glcm_np(img, vmin=0, vmax=255, nbit=8, kernel_size=5):
    mi, ma = vmin, vmax
    ks = kernel_size
    h,w = img.shape

    # digitize
    bins = np.linspace(mi, ma+1, nbit+1)
    gl1 = np.digitize(img, bins) - 1
    gl2 = np.append(gl1[:,1:], gl1[:,-1:], axis=1)

    # make glcm
    glcm = np.zeros((nbit, nbit, h, w), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            mask = ((gl1==i) & (gl2==j))
            glcm[i,j, mask] = 1

    kernel = np.ones((ks, ks), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            glcm[i,j] = cv2.filter2D(glcm[i,j], -1, kernel)

    glcm = glcm.astype(np.float32)
    return glcm

def get_glcm_mean_np(glcm):
    '''
    calc glcm mean
    '''
    nbit, _, h, w = glcm.shape
    mean = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i,j] * i / (nbit)**2
    return mean

def get_glcm_std_np(glcm):
    '''
    calc glcm std
    '''
    nbit, _, h, w = glcm.shape
    mean = get_glcm_mean_np(glcm)

    std2 = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            std2 += (glcm[i,j] * i - mean)**2

    std = np.sqrt(std2)
    return std


def get_glcm_contrast_np(glcm):
    '''
    calc glcm contrast
    '''
    nbit, _, h, w = glcm.shape
    cont = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            cont += glcm[i,j] * (i-j)**2
    return cont


def get_glcm_dissimilarity_np(glcm):
    '''
    calc glcm dissimilarity
    '''
    nbit, _, h, w = glcm.shape
    diss = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            diss += glcm[i,j] * np.abs(i-j)
    return diss


def get_glcm_homogeneity_np(glcm):
    '''
    calc glcm homogeneity
    '''
    nbit, _, h, w = glcm.shape
    homo = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            homo += glcm[i,j] / (1.+(i-j)**2)
    return homo


def get_glcm_ASM_np(glcm):
    '''
    calc glcm asm, energy
    '''
    nbit, _, h, w = glcm.shape
    asm = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            asm  += glcm[i,j]**2

    ene = np.sqrt(asm)
    return asm, ene


def get_glcm_max_np(glcm):
    '''
    calc glcm max
    '''
    max_  = np.max(glcm, axis=(0,1))
    return max_


def get_glcm_entropy_np(glcm, ks=5):
    '''
    calc glcm entropy
    '''
    pnorm = glcm / np.sum(glcm, axis=(0,1)) + 1./ks**2
    ent  = np.sum(-pnorm * np.log(pnorm), axis=(0,1))
    return ent

if __name__ == "__main__":
    main()