import streamlit as st 
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt
import cv2 as cv
#streamlit run test.py



#Functions---------------------------------------
def inversion(image):
   image = np.array(image.convert('L'))
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
   fig = plt.figure()
   plt.xlabel("Pixel values")
   plt.ylabel("Frequency (number of pixels)")
   plt.title("Histogram before inversion")
   plt.plot(H1,color='blue')
   #display inverted image
   st.pyplot(fig)
   st.write("Inverted image : ")
   st.image(img)
   #Histogram after inversion       
   fig1 = plt.figure()
   plt.xlabel("Pixel values")
   plt.ylabel("Frequency (number of pixels)")
   plt.title("Histogram of inverted img")
   plt.plot(H2,color='red')
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
def expansion(image):
   image = np.array(image.convert('L'))
   h,w= image.shape
   valeurs = image.ravel()
   min_val = min(valeurs)
   max_val = max(valeurs)  
   img_res = np.zeros(image.shape,np.uint8)
   for i in range(h):
       for j in range(w): 
           img_res[i,j]=(255*(image[i,j]-min_val))/(max_val-min_val)   
   st.write("Image after expansion :")
   st.image(img_res)

def translation (image,ecart):
  image = np.array(image.convert('L'))
  h,w= image.shape
  His = np.zeros((256,),np.uint32)
  for i in range(h):
    for j in range(w): 
        His[image[i,j]]+=1
  image_translated = image+ecart
  st.write("Image after translation :")
  st.image(image_translated)
  histogram_translated, _ = np.histogram(image_translated.flatten(), bins=256, range=[0, 256])
   # Plot the histogram
  fig = plt.figure(figsize=(8, 4))
  plt.bar(range(256), histogram_translated, color="green")
  plt.xlabel('Pixel Value')
  plt.ylabel('Frequency')
  plt.title('Histogram after Translation')
  plt.xlim(np.min(image_translated), np.max(image_translated))
  #or plt.xlim(0, 255)
  plt.grid(True)
  # Display the histogram plot using Streamlit
  st.write("Histogram after translation:")
  st.pyplot(fig)

def egalisation(image):
   image = np.array(image.convert('L'))
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
   st.pyplot(fig)
   
#sideBar---------------------------------------------------------

with st.sidebar:
    #File uploader
    uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    selectbox = st.selectbox(
    "Select a method :",
    ("","Inversion","Expansion_dynamique","Translation","Egalisation", "Quantification","")
    )
    if selectbox=="Quantification":
       select_quantification = st.selectbox("Select a quantification method :",("","Uniforme","Median_cut","Classification"))
      



if uploaded_image is not None:
  image = Image.open(uploaded_image)

  st.write("The uploaded image :")
  st.image(image, caption='Original Image', use_column_width=True)
  gris(image)
   #if selectbox is not None and inverted image is not None:
  if selectbox =="Inversion":
    inversion(image)
  elif selectbox =="Expansion_dynamique":
     expansion(image)
  elif selectbox =="Translation":
     ecart = st.sidebar.slider("Translation Value", -255, 255, 0)
     translation(image,ecart)  #try to use slider in app
  elif selectbox =="Egalisation":
     egalisation(image)
     
    

