# -*- coding: utf-8 -*-
import streamlit as st
from skimage import io, exposure, img_as_ubyte, filters, morphology, color
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

st.title("Final Project")
st.write("Rihhadatul Aisy Nadhilah")
st.write("5023211020")

# Load gambar langsung dari path lokal
image_path = r"C:/Users/Lenovo/OneDrive - Institut Teknologi Sepuluh Nopember/Cooleyeah/Semester 7/PCM/Bu Nada/Dataset-20240910/mri.jpg"

# Baca gambar
image = io.imread(image_path)

# Konversi gambar ke grayscale jika gambar dalam RGB
if image.ndim == 3:  # jika gambar memiliki 3 channel
    image = color.rgb2gray(image)  # konversi ke grayscale

st.header("Original Brain MRI")
st.image(image, caption="Loaded MRI Image", width=400)

# Adaptive Histogram Equalization
image_adapteq = exposure.equalize_adapthist(image, clip_limit=0.01)
image_adapteq = img_as_ubyte(image_adapteq)

st.subheader("Histograms of Original and AHE Images")

# Membuat dua kolom
col1, col2 = st.columns(2)

# Kolom 1 untuk Histogram Gambar Asli
with col1:
    hist_original, bin_edges = exposure.histogram(image)
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    ax1.plot(bin_edges[1:], hist_original, color='orange')
    ax1.set_title("Histogram of Original Image")
    ax1.set_xlabel("Pixel Intensity")
    ax1.set_ylabel("Frequency")
    st.pyplot(fig1)

# Kolom 2 untuk Histogram AHE Image
with col2:
    hist_ahe, bin_edges = exposure.histogram(image_adapteq)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.plot(bin_edges[1:], hist_ahe, color='blue')
    ax2.set_title("Histogram of AHE Image")
    ax2.set_xlabel("Pixel Intensity")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)

# Normalisasi gambar AHE ke rentang [0.0, 1.0]
my_gray_norm = image_adapteq / 255.0

# Tampilkan gambar di Streamlit
st.header("Grayscale Image with Adaptive Histogram Equalization")
st.image(my_gray_norm, caption="Grayscale Image with AHE", width=400)

# Median filter
med_img = ndi.median_filter(image, size=5)
med_img_norm = med_img / 255.0
st.header("Image after Median Filter")
st.image(med_img_norm, caption="Image after Median Filter", width=400)

# Otsu Thresholding
threshold_value = filters.threshold_otsu(med_img)
mask = med_img > threshold_value
mask_float = mask.astype(float)
st.header("Masking Image")
st.image(mask_float, caption=f"Masked Image (Threshold: {threshold_value})", width=400)

# Convert mask to int untuk operasi lebih lanjut
im = np.where(mask, med_img, 0)
hist_im, bin_edges = exposure.histogram(im)

# Membuat dua kolom untuk menampilkan Masked Image
col1, col2 = st.columns(2)
with col1:
    st.image(mask_float, caption=f"Masked Image (Threshold: {threshold_value})", use_column_width=True)
with col2:
    im_norm = im / 255.0
    st.image(im_norm, caption="Masked Region", use_column_width=True)

# Histogram Masked Image
st.subheader("Histogram of Masked Image")
fig, ax = plt.subplots()
ax.plot(bin_edges[1:], hist_im)
ax.set_title("Histogram of Masked Image")
ax.set_xlabel("Pixel Intensity")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Mask tuning with erosion
mask_erose = ndi.binary_erosion(mask, iterations=5)
mask_erose_float = mask_erose.astype(float)
st.header("Tuning Erosion")
st.image(mask_erose_float, caption="Mask after Erosion", width=400)
    
# Remove small objects
cleaned_mask = morphology.remove_small_objects(mask_erose, min_size=1500)
cleaned_mask_float = cleaned_mask.astype(float)
st.header("Segmentation Image")
st.image(cleaned_mask_float, caption="Cleaned Mask (Central Region)", width=400)

# Segment the central region
segmented_region = med_img_norm * cleaned_mask_float
st.image(segmented_region, caption="Segmented Central Region (Final)", width=400)

# Display Original and Segmented Images side by side
st.header("Original and Final Segmented Images")
col1, col2 = st.columns(2)
with col1:
    st.image(image, caption="Original MRI Image", use_column_width=True)
with col2:
    st.image(segmented_region, caption="Segmented Central Region (Final)", use_column_width=True)
