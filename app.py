import streamlit as st 
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np
@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('model_cnn.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Malaria Detection
         """
         )

file = st.file_uploader("Please upload the image of Blood cell", type=["jpg", "png"])
def import_and_predict(image_data, model):
    
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
    
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    a=np.argmax(predictions, axis=1)
    if(a==1):
      st.write(" The Blood is Uninfected")
    else:
      st.write(" The Blood is Infected")
    
    