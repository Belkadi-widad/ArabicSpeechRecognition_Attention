#!/usr/bin/env python
# coding: utf-8

# # Faire des prédictions 

# In[1]:


from google.colab import drive 
drive.mount('/content/drive')


# In[2]:


import sys
sys.path.append('/content/drive/My Drive/Colab Notebooks/SpeechRecognition')
get_ipython().system('pip install soundfile')
get_ipython().system('pip install tf-nightly')
get_ipython().system('pip install tensorflow-gpu')


# In[3]:


sys.path.append('/content/drive/My Drive/Colab Notebooks/SpeechRecognition')

import tensorflow as tf
#import Preprocess_SpeechReco as old_data
import Preprocess_bigdataset as data 
import os 
from tensorflow.keras.models import Sequential, Model , load_model
import numpy as np
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
import time 
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D


# In[4]:


x_train,y_train,x_test,y_test,dic_voca=data.getData(mfccs=True,split=True)


# In[5]:


x_train_no_mfcc,_,x_test_no_mfcc,_,_=data.getData(mfccs=False,split=True)


# In[6]:


import numpy as np


def beam_search(model,  k, src_input, sequence_max_len,dict_voc,mfcc):
    # (log(1), initialize_of_zeros)
    if mfcc: 
      src_input=np.reshape(src_input, (1,src_input.shape[0],src_input.shape[1]) )

    else : 
      src_input=np.reshape(src_input, (1,src_input.shape[0]) )
    k_beam = [(0, [float(dict_voc['@'])]*(sequence_max_len))]
    print(k_beam)
    
    # l : point on target sentence to predict
    for l in range(sequence_max_len):
        all_k_beams = []
        for prob, sent_predict in k_beam:
            predicted = model.predict([src_input, np.array([sent_predict])])[0]
            #print(predicted)
            # top k!
            possible_k = predicted[l].argsort()[-k:][::-1]
            #print(possible_k)
            # add to all possible candidates for k-beams
            all_k_beams += [
                (
                    sum(np.log(predicted[i][sent_predict[i+1]]) for i in range(l)) + np.log(predicted[l][next_wid]),
                    list(sent_predict[:l+1])+[next_wid]+[0]*(sequence_max_len-l-2)
                )
                for next_wid in possible_k
            ]
        # top k
        k_beam = sorted(all_k_beams)[-k:]

    return k_beam


# 

# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import seaborn



def attent_and_generate( preprocessd_input_audio,dict_voc , OUTPUT_LENGTH):
    seaborn.set(font=['Osaka'], font_scale=3)

    preprocessd_input_audio=np.reshape(preprocessd_input_audio, (1,preprocessd_input_audio.shape[0]) )
    print(preprocessd_input_audio[0][100])
    START_CHAR_CODE = dict_voc['@']
    decoder_input = np.zeros(shape=(len(preprocessd_input_audio), OUTPUT_LENGTH))
    decoder_input[:,0] = START_CHAR_CODE
   
    for i in range(1, OUTPUT_LENGTH):
        
        #argmax donne l'indice qui a la plus grande proba !
        output, attention=attention_model.predict([preprocessd_input_audio, decoder_input])
        output = output.argmax(axis=2)        
        decoder_input[:,i] = output[:,i]
        attention_density = attention[0]
        decoded_output = data.Decode_sentence(decoder_input[0][1:],"DICT",dict_voc)
        
    return decoder_input[:,1:], decoded_output

def visualize(text,dict_voc , OUTPUT_LENGTH):
    model= load_model("/content/drive/My Drive/Colab Notebooks/SpeechRecognition/fitted_moddels/Modele_SpeechReco_1598612777_[0.3621765375137329, 0.8927567601203918].h5", 
                      custom_objects={'Melspectrogram':Melspectrogram,
                            'Normalization2D':Normalization2D})
    attention_layer = model.get_layer("tf_op_layer_concat_8") # or model.get_layer("tf_op_layer_concat_8")
    attention_model = Model(inputs=model.inputs, outputs=model.outputs + [attention_layer.output])
    attention_model.summary()
    attention_density, katakana = attent_and_generate(text,dict_voc , OUTPUT_LENGTH)
    print("visualize chart " , katakana)
    plt.clf()
    plt.figure(figsize=(148,120))

    ax = seaborn.heatmap(attention_density[:len(katakana) + 2, : len(text) + 2],
        xticklabels=[w for w in range(0,12)],
        yticklabels=[w for w in katakana])

    ax.invert_yaxis()
    plt.show()
#visualize(x_train[60],dic_voca,y_train.shape[1]-1)


# In[ ]:


def predict(k, path_model , path_audio,tokenzer_train , OUTPUT_LENGTH,mfccs): 
    
    #get in the model  
    model= load_model(path_model, 
                  custom_objects={'Melspectrogram':Melspectrogram,
                        'Normalization2D':Normalization2D})
    
    #preprocessd_input_audio=preprocess_new_audio(path_audio,mfcc=mfccs)[0]
    preprocessd_input_audio=path_audio
    print(preprocessd_input_audio.shape)
    #preprocessd_input_audio=path_audio
    ''' if k==1:
        predictions = Greedy_search(model, preprocessd_input_audio,tokenzer_train , OUTPUT_LENGTH)'''
  
    predictions =beam_search(model,k,preprocessd_input_audio,OUTPUT_LENGTH, dic_voca,mfcc=mfccs)
    print(predictions)
        
    return predictions

def preprocess_new_audio(path_audio="/content/drive/My Drive/Colab Notebooks/SpeechRecognition/s8L2.wav",max_duration=12,mfcc=False) :

  audio,sr= data.rosa.load(path_audio)
  print(sr)
  dur_audio=data.rosa.get_duration(audio)
  audios=[]

  if dur_audio<=max_duration : 
    #si inférieur à la durée le pad 
    pad_ms=(max_duration-dur_audio) *sr
    print("old duration ", dur_audio )
    audio=np.append(audio,np.zeros(int(round(pad_ms)))) 
    data.Plot_audio(audio,sr)

    if mfcc : 
        audio=data.build_mfccs([audio])[0]
    audios.append(audio)
  else : 
    
    # extraire les sous audios 
    nonMuteSections =data.rosa.effects.split(audio,60)
    for i in range(0,len(nonMuteSections)):  
      s=audio[nonMuteSections[i][0]:nonMuteSections[i][1]]
      if mfcc : 
        s=data.build_mfccs([s])
      audios.append(s)  
  return audios    



#path_model="/content/drive/My Drive/Colab Notebooks/SpeechRecognition/fitted_moddels/Modele_SpeechReco_1598612777_[0.3621765375137329, 0.8927567601203918].h5" 
path_model="/content/drive/My Drive/Colab Notebooks/SpeechRecognition/fitted_moddels/Modele_SpeechReco_1599299782_[0.3388335406780243, 0.9021098017692566].h5"
#path_model="/content/drive/My Drive/Colab Notebooks/SpeechRecognition/fitted_moddels/Modele_SpeechReco_1600302374_[0.35865363478660583, 0.896904468536377].h5"
#path_audio="/content/drive/My Drive/Colab Notebooks/SpeechRecognition/output.wav"
index=754   
k=64
predictions=predict(k ,path_model,x_test[index],None,y_test.shape[1]-1,True)


# In[ ]:


def viewPredictions(text_true=u' ' , predictions=[],k=k):

  for el in range(0,k) : 
    predicted = data.Decode_sentence(predictions[el][1],"DICT",dic_voca)
    proba=predictions[el][0]
    
    print(" output beam :  " ,proba,"\n ", predicted , ' \n', data.Test_Transcription( predicted[1] , False ))

  true =data.Decode_sentence(y_test[index],"DICT",dic_voca)
  print(" le true ", true)   
  true =data.Test_Transcription( true[1] , False )
  print(" le true ", true) 
data.Plot_audio(x_train_no_mfcc[index],22050)
viewPredictions(text_true="العالمية",predictions=predictions)


# In[ ]:




