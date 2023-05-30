import streamlit as st 
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import median
#streamlit run test.py



#Functions---------------------------------------
def histogram1(image):
   image = np.array(image.convert('L'))
   h,w= image.shape
   #creation du premier histogramme
   H1 = np.zeros((256,1),np.uint32)
   for i in range(h):
    for j in range(w):    
            H1[image[i,j]]+=1              
   fig = plt.figure()
   plt.xlabel("Pixel values")
   plt.ylabel("Frequency (number of pixels)")
   plt.title("Histogram")
   plt.plot(H1,color='blue')
   #display inverted image
   st.pyplot(fig)
   
def inversion(image1):
   image = np.array(image1.convert('L'))
   h,w= image.shape
   img = np.zeros(image.shape,np.uint8)
   #creation du premier histogramme
   H1 = np.zeros((256,1),np.uint32)
   H2 = np.zeros((256,1),np.uint32)
   for i in range(h):
    for j in range(w):    
            img[i,j] = 255-image[i, j]
            H1[image[i,j]]+=1
            H2[img[i,j]]+=1     
   #Histogram before inversion
   histogram1(image1)
   #display inverted image
   st.write("Inverted image : ")
   st.image(img)
   #Histogram after inversion       
   fig1 = plt.figure()
   plt.xlabel("Pixel values")
   plt.ylabel("Frequency (number of pixels)")
   plt.title("Histogram inverted")
   plt.plot(H2,color='red')
   st.write("Histogram after inversion :")
   st.pyplot(fig1)

def gris(image):
  image=np.array(image)
  h,w,c = image.shape
  img2 = np.zeros(image.shape,np.uint8)

  for i in range(h):
    for j in range(w):
        b,g,r = image[i,j]
        img2[i,j] = 0.2989*r + 0.5870*g + 0.1140*b
  st.write("GrayScaled image :")
  st.image(img2)

#def egalisation(image):
def expansion(image1):
   image = np.array(image1.convert('L'))
   h,w= image.shape
   valeurs = image.ravel()
   min_val = min(valeurs)
   max_val = max(valeurs)  
   img_res = np.zeros(image.shape,np.uint8)
   H2 = np.zeros((256,1),np.uint32)
   for i in range(h):
       for j in range(w): 
           img_res[i,j]=(255*(image[i,j]-min_val))/(max_val-min_val) 
           H2[img_res[i,j]]+=1  
   st.write('histogram before expansion :')
   histogram1(image1)
   st.write("Image after expansion :")
   st.image(img_res)
   #plot hist after expansion
   fig1 = plt.figure()
   plt.xlabel("Pixel values")
   plt.ylabel("Frequency (number of pixels)")
   plt.title("Histogram")
   plt.plot(H2,color='orange')
   st.write("Histogram after expansion :")
   st.pyplot(fig1)

def translation (image1,ecart):
  image = np.array(image1.convert('L'))
  image_translated = image+ecart
  st.write("Image after translation :")
  st.image(image_translated)
  histogram_translated, _ = np.histogram(image_translated.flatten(), bins=256, range=[0, 256])
   # Plot the histogram
  fig = plt.figure(figsize=(8, 4))
  plt.bar(range(256), histogram_translated, color="green")
  plt.xlabel('Pixel Value')
  plt.ylabel('Frequency')
  plt.title('Histogram ')
  plt.xlim(0, 255)
  plt.grid(True)
  # Display the histogram plot using Streamlit
  st.write("Histogram after translation:")
  st.pyplot(fig)

def egalisation(image1):
   image = np.array(image1.convert('L'))
   h,w= image.shape
   #histgramme 1
   H1 = np.zeros((256,1),np.uint32)
   for i in range(h):
      for j in range(w): 
          H1[image[i,j]]+=1
   H_cn = np.cumsum(H1)
   H_cn=H_cn/h*w
   valeurs = image.ravel()
   max_val = max(valeurs)
   H_eg=np.copy(H_cn)*max_val 
   img_res = np.zeros(image.shape,np.uint8) 
   for i in range(h):
       for j in range(w): 
           img_res[i,j]=H_eg[image[i,j]]
   st.write("Histogram before egalisation :")
   histogram1(image1)
   st.write("Image after egalisation :")
   st.image(img_res)
   #Deuxième histogramme
   H2 = np.zeros((256,1),np.uint32)
   for i in range(h):
       for j in range(w): 
           H2[img_res[i,j]]+=1        
   #plot histogramme 2        
   fig = plt.figure()
   plt.xlabel("Pixel values")
   plt.ylabel("Frequency (number of pixels)")
   plt.title("Histogram")
   plt.plot(H2,color='red')
   #Histogram after egalisation
   st.write("Histogram after egalisation")
   st.pyplot(fig)
   
def detect_contours(image):
    image_np = np.array(image)
    # Conversion en niveaux de gris
    gray_image = cv.cvtColor(image_np, cv.COLOR_RGB2GRAY)
    # Application du filtre de gradient de Sobel
    sobel_x = cv.Sobel(gray_image, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(gray_image, cv.CV_64F, 0, 1, ksize=3)
    # Calcul du module du gradient
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    # Normalisation des valeurs entre 0 et 255
    gradient_magnitude = cv.normalize(gradient_magnitude, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

    return gradient_magnitude

def segmentation_par_regions(image, seuil):
    image = np.array(image.convert('L'))
    h, w = image.shape 
    # Création d'une image binaire pour la segmentation
    segmentation = np.zeros((h, w), dtype=np.uint8) 
    # Parcours de l'image pour la segmentation
    for i in range(h):
        for j in range(w):
            # Comparaison de la valeur du pixel avec le seuil
            if image[i, j] > seuil:
                segmentation[i, j] = 255  # Assigner une valeur blanche aux pixels supérieurs au seuil 
    return segmentation

#quantification uniforme et classification--------------

def select_initial_colors(pixels, num_colors):
    initial_colors = np.zeros((num_colors, 3), dtype=np.uint8)
    
    # Sélectionner les couleurs initiales en fonction des valeurs extrêmes des canaux RGB
    for i in range(3):
        min_val = np.min(pixels[:, i])
        max_val = np.max(pixels[:, i])
        initial_colors[:, i] = np.linspace(min_val, max_val, num_colors)
    
    return initial_colors

def split_palette(pixels, palette, colors, level):
    if level >= len(colors):
        return
    
    indices = np.where(np.all(pixels == colors[level], axis=1))[0]
    median = np.median(indices)
    palette[level] = colors[level]
    
    split_palette(pixels[:median], palette, colors, level + 1)
    split_palette(pixels[median:], palette, colors, level + 1)

def apply_palette(image, palette):
    h, w, _ = image.shape
    quantized_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i in range(h):
        for j in range(w):
            pixel = image[i, j]
            closest_color = find_closest_color(pixel, palette)
            quantized_image[i, j] = closest_color
    
    return quantized_image

def find_closest_color(pixel, palette):
    distances = np.linalg.norm(palette - pixel, axis=1)
    closest_index = np.argmin(distances)
    closest_color = palette[closest_index]
    
    return closest_color
def uniform_quantization(image, num_colors):
    image = np.array(image)
    h, w, _ = image.shape
    
    # Diviser l'espace des couleurs uniformément
    step = 256 // num_colors
    palette = np.zeros((num_colors, 3), dtype=np.uint8)
    for i in range(num_colors):
        palette[i] = [i * step, i * step, i * step]
    
    # Conversion des pixels de l'image vers les couleurs de la palette
    quantized_image = apply_palette(image, palette)
    
    return quantized_image


def classification_quantization(image, num_colors):
    image = np.array(image)
    h, w, _ = image.shape
    
    # Sélectionner aléatoirement des couleurs initiales pour la palette
    np.random.seed(0)
    palette = np.random.randint(0, 256, size=(num_colors, 3), dtype=np.uint8)
    
    # Conversion des pixels de l'image vers les couleurs de la palette
    quantized_image = apply_palette(image, palette)
    
    return quantized_image


#sideBar---------------------------------------------------------
st.title("Image processing")
st.subheader("Platforme d'application des différentes fonctionnalités de traitement d'images")
with st.sidebar:
    #File uploader
    uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    selectbox = st.selectbox(
    "Select a method :",
    ("","Inversion","Expansion dynamique","Translation","Egalisation", "Quantification","Segmentation")
    )
    if selectbox=="Quantification":
       select_quantification = st.selectbox("Select a quantification method :",("","Uniforme","Classification"))
    if selectbox=="Segmentation":
       selectbox_seg = st.selectbox("Select a segmentation method :",("","Detection Contours","Segmentation region"))



if uploaded_image is not None:
  image = Image.open(uploaded_image)

  st.write("The uploaded image :")
  st.image(image, caption='Original Image', use_column_width=True)
  gris(image)
   #if selectbox is not None and inverted image is not None:
  if selectbox =="Inversion":
    inversion(image)
  elif selectbox =="Expansion dynamique":
     expansion(image)
  elif selectbox =="Translation":
     ecart = st.sidebar.slider("Translation Value", -255, 255, 0)
     translation(image,ecart)  #try to use slider in app
  elif selectbox =="Egalisation":
     egalisation(image) 
  elif selectbox=="Segmentation":
    if selectbox_seg == "Segmentation region":
      seuil = st.sidebar.slider("Threshold", 0, 255, 128)
      segmented_image = segmentation_par_regions(image, seuil)
      st.write("Segmented Image:")
      st.image(segmented_image)
    elif selectbox_seg =="Detection Contours":
      contour_image = detect_contours(image)
      st.write("Image with detected contours:")
      st.image(contour_image, caption='Contour Image', use_column_width=True)

  elif selectbox =="Quantification":
    if select_quantification=="Uniforme":
      num_colors = st.slider("Number of Colors", 1, 256)
      quantized_image = uniform_quantization(image, num_colors)
      st.write("Image after Uniform Quantization:")
      st.image(quantized_image)
    elif select_quantification == "Classification":
        num_colors = st.slider("Number of Colors", 1, 256,)
        quantized_image = classification_quantization(image, num_colors)
        st.write("Image after Classification Quantization:")
        st.image(quantized_image)
   
    

