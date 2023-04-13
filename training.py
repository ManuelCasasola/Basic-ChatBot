import random
import json
import pickle
import numpy as np
import nltk


from keras.models import Sequential
from nltk.stem import WordNetLemmatizer
from keras.layers import Dense, Activation,Dropout
from keras.optimizers import SGD


lematizador=WordNetLemmatizer()

#Lee el archivo .json
predictions=json.loads(open("predictions.json").read())

#Creamos listas vacías para almacenar la informacion
palabras=[]
clases=[]
documentos=[]
ignorar_signos=["/","!","#"]
for prediction in predictions["predictions"]:
    for patron in prediction['patrones']:
        #separamos las palabras por patrones
        palabra_lista=nltk.word_tokenize(patron)
        palabras.extend(palabra_lista)# Y la añadimos a la lista de palabras que creamos con anterioridad.

        #asociamos los patrones con las respectivas etiquetas
        documentos.append(((palabra_lista),prediction['etiqueta']))

        #Añadimos las etiquetas en la lista de clases
        if prediction['etiqueta'] not in clases:
            clases.append(prediction['etiqueta'])


#Guardamos las palabras guardadas en el lematizador.
palabras=[lematizador.lemmatize(palabra)
          for palabra in palabras if palabra not in ignorar_signos]
palabras=sorted(set(palabras))

#Guardamos las palabras y clases en archivos binarios
pickle.dump(palabras,open('palabras.pkl','wb'))
pickle.dump(clases,open('clases.pkl','wb'))

#Ahora, necesitareos transformar 
# las palabras en valores numericos 
#ya que la red neurona trabaja con 
# valores numericos
training=[]
almacenaje_salida=[0]*len(clases)
for documento in documentos:
    bag=[]
    patrones_palabras=documento[0]
    patrones_palabras=[lematizador.lemmatize(
        palabra.lower()) for palabra in patrones_palabras]
    for palabra in palabras:
        bag.append(1) if palabra in patrones_palabras else bag.append(0)
    
    #Hacemos una copia para el almacenaje_salida
    output_row=list(almacenaje_salida)
    output_row[clases.index(documento[1])]=1
    training.append([bag,output_row])
random.shuffle(training)
training=np.array(training)

#Dividimos la informacion
train_x=list(training[:,0])
train_y=list(training[:,1])

#Creamos el Modelo Secuencial de Machine Learning
modelo=Sequential()
modelo.add(Dense(128,input_shape=(len(train_x[0]),),
                 activation='relu'))
modelo.add(Dropout(0.5))
modelo.add(Dense(64,activation='relu'))
modelo.add(Dropout(0.5))
modelo.add(Dense(len(train_y[0]),
                 activation='softmax'))

#Compilamos el modelo
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
modelo.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])
hist = modelo.fit(np.array(train_x), np.array(train_y),
                 epochs=200, batch_size=5, verbose=1)

#Guardamos el modelo
modelo.save('chatbotmodel.h5',hist)

#Printeamos un mensaje para ver si funciona todo correctamente
print("ChatBot Entrenado")

