#from phlatib import Path
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from io import BytesIO




#def predict(image):
def predict(path):

 
    model_path = "./logdir/model_file.hdf5"
   

    classes = ["ムロツヨシ","阿部寛","綾瀬はるか","吉沢亮","吉田沙保里","橋本環奈","窪田正孝","広瀬すず","佐藤健","佐藤二朗","山崎健斗","篠原涼子","菅田将暉","石原さとみ","大泉洋","二階堂ふみ","北川景子","本田翼","木村拓哉","有村架純"]
    # load model
    model = load_model(model_path)

    
    image_size=64

    X = []
    
    image=path
    
    image=Image.open(image)
  
    image = image.convert("RGB")
    image = image.resize((image_size, image_size))
    data = np.asarray(image)
    X.append(data)
    X = np.array(X)
    #正規化(0-1)
    X = X.astype('float32')
    X = X / 255.0

    result = model.predict([X])[0]
    predicted = result.argmax()
    percentage = int(result[predicted] * 100)
    return "{0}({1} %)".format(classes[predicted],percentage)






   
    
#print('ok')
#print(predict('./sample_image/yosida.jpg'))

