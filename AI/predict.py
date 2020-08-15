#from phlatib import Path
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from io import BytesIO


#####  mike=0,siro=1  ######


def predict(image):

    #pil_img = Image.open(image)#PILで読み込む
    #img=BytesIO()#空のインスタンスを作る
    #pil_img.save(img,'jpeg')#さっきのインスタンスに保存

    #model_path = "../../original_aug/logdir/model_file.hdf5"
    model_path = "./logdir/model_file.hdf5"
    #classes = ["mike","siro"]

    classes = ["ムロツヨシ","阿部寛","綾瀬はるか","吉沢亮","吉田沙保里","橋本環奈","窪田正孝","広瀬すず","佐藤健","佐藤二朗","山崎健斗","篠原涼子","菅田将暉","石原さとみ","大泉洋","二階堂ふみ","北川景子","本田翼","木村拓哉","有村架純"]
    # load model
    model = load_model(model_path)

    #image_size=100
    image_size=64

    X = []
    
    image=Image.open(image)
    #image = Image.open(pil_img)
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


    #if result[predicted]>0.25:
    #    return "This is {}".format(classes[predicted])
    #else:
    #    return "Cat is not exist"




   
    
#print('ok')
#print(predict('./sample_image/yosida.jpg'))

