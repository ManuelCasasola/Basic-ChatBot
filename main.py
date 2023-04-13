import random
import json
import pickle 
import numpy as np
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer


lematizador=WordNetLemmatizer()

#Cargamos los archivos que generamos con anterioridad
predicciones=json.loads(open("predictions.json").read())
palabras=pickle.load(open('palabras.pkl','rb'))
clases=pickle.load(open('clases.pkl','rb'))
modelo=load_model('chatbotmodel.h5')

#Definimos 4 funciones

#1- Esta función separa las palabras de las frase o frases insertada
def separa_palabras(frase):
    palabras_frase=nltk.word_tokenize(frase)
    palabras_frase=[lematizador.lemmatize(palabra)
                    for palabra in palabras_frase]
    return palabras_frase

#2-Esta función encontrara si la palabra se encuentra en la frase
def bagw(frase):
    palabras_frase=separa_palabras(frase)
    bag=[0]*len(palabras)
    for w in palabras_frase:
        #Y la guardamos en la variable creada para guardar las palabras
        for i, palabra in enumerate(palabras):
            if palabra==w:
                bag[i]=1
    return np.array(bag)

#3-Esta función predicirá la clase de la frase
def prediccion_clase(frase):
    bow=bagw(frase)
    res=modelo.predict(np.array([bow]))[0]
    ERROR_THRESHOLD=0.25
    resultados=[[i,r] for i, r in enumerate(res)
                if r > ERROR_THRESHOLD]
    resultados.sort(key=lambda x: x[1], reverse=True)
    return_list=[]
    for r in resultados:
        return_list.append({'prediccion':clases[r[0]],
                            'probability':str(r[1])})
        return return_list

#4-Esta ultima funcion
#generara una respuesta random.

def dar_respuesta(predicciones_list,predicciones_json):
    etiqueta=predicciones_list[0]['prediccion']
    lista_predicciones=predicciones_json['predictions']
    resultado=""
    for i in lista_predicciones:
        if i['etiqueta']==etiqueta:
            resultado=random.choice(i['respuestas'])
            break
    return resultado

print("Chatbot Activado")

while True:
    mensaje=input("")
    pred=prediccion_clase(mensaje)
    res=dar_respuesta(pred,predicciones)
    print(res)