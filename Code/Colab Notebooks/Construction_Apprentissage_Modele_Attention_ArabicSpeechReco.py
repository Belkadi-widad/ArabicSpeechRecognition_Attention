#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# manter le drive 
from google.colab import drive 
drive.mount('/content/drive')


# In[ ]:


#instalation des package utiles 
import sys
sys.path.append('/content/drive/My Drive/Colab Notebooks/SpeechRecognition')
get_ipython().system('pip install soundfile')
get_ipython().system('pip install tf-nightly')
get_ipython().system('pip install tensorflow-gpu')
get_ipython().system('pip install -U tensorboard_plugin_profile')


# In[ ]:


#importer les p
import tensorflow as tf
import librosa
#import Preprocess_SpeechReco as data
import Preprocess_bigdataset as data 
import kapre
import os 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Lambda, Permute ,Conv2D,BatchNormalization
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
import numpy as np
from tensorflow.keras import layers as L
from tensorflow.keras import backend as K
from tensorflow.compat.v1.keras.layers import CuDNNLSTM, CuDNNGRU
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler,ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
import time 
import pandas as pd
from sklearn.model_selection import StratifiedKFold,KFold
from contextlib import redirect_stdout


# # Fonctions utiles pour la construction de la structure du RDN 

# In[ ]:


'''
Cette fonction permet de charger les données à partir du drive et de les préparer pour l apprentissage 
'''
def Load_resempaled_data(shuffle=True ,split=False,mfccs=False): 

  if split==True :
    #no cross validation 

    x_train,y_train,x_test,y_test,dict_voc= data.getData(split=True,mfccs=mfccs)
    training_encoder_input=x_train
    training_decoder_output=y_train
    training_decoder_input = np.zeros_like(training_decoder_output)
    training_decoder_input = training_decoder_output[:,:-1]
    voc_size=len(dict_voc)
    output_length=training_decoder_input.shape[1]
    input_length=training_encoder_input.shape[1]
    #one hot incoding 
    print(training_decoder_output[:,1:])
    training_decoder_output = np.eye(voc_size)[training_decoder_output[:,1:].astype('int')]
    print(training_decoder_input)
    print(training_decoder_input.shape)
    print(training_decoder_output.shape)
    validation_encoder_input=x_test 
    validation_decoder_output=y_test 
    print(validation_decoder_output.shape)
    validation_decoder_input = np.zeros_like(validation_decoder_output)
    validation_decoder_input = validation_decoder_output[:,:-1]
    #one hot incoding 
    validation_decoder_output = np.eye(voc_size)[validation_decoder_output[:,1:].astype('int')]
    print(validation_decoder_input.shape)
    print(validation_decoder_output.shape)
    

    return   input_length,output_length,voc_size,training_encoder_input, training_decoder_input,training_decoder_output,validation_encoder_input, validation_decoder_input, validation_decoder_output
  else : 
    #for cross validation 
    
    x_train,y_train,dict_voca= data.getData(split=False,mfccs=mfccs)
    print(x_train.shape)
    print(y_train.shape)
    dict_voc=dict_voca
    training_encoder_input=x_train
    #training_decoder_output=y_train[3083:]
    training_decoder_output=y_train
    training_decoder_input = np.zeros_like(training_decoder_output)
    training_decoder_input = training_decoder_output[:,:-1]
    voc_size=len(dict_voc)
    output_length=training_decoder_input.shape[1]
    input_length=training_encoder_input.shape[1]
    #one hot incoding 
    print(training_decoder_output[:,1:])
    training_decoder_output = np.eye(voc_size)[training_decoder_output[:,1:].astype('int')]
    print(training_decoder_input)
    print(training_decoder_input.shape)
    print(training_decoder_output.shape)

    return   input_length,output_length,voc_size,training_encoder_input, training_decoder_input,training_decoder_output


# In[ ]:


#for the encoder 
# cette fonction permet de construire la couche qui produit les spectrogrammes de mel ( au lieu  des mfccs )
def Build_MelSpectrogram(Parametres_layer,input_length): 

  mel_layer = Melspectrogram(n_dft=Parametres_layer["n_dft"],
                             n_hop=Parametres_layer["n_hop"],
                             input_shape= (1,input_length),
                             padding=Parametres_layer["padding"], 
                             sr= Parametres_layer["sr"], 
                             n_mels=Parametres_layer["n_mels"],
                             fmin=40.0, fmax=  Parametres_layer["sr"] / 2, power_melgram=1.0,
                             return_decibel_melgram=True, trainable_fb=False,
                             trainable_kernel=False,
                             name='mel_stft')
  mel_layer.trainable = False
  
  return mel_layer

# construire une couche BRNN 

def Build_Bidiractionnel_layer(Parametres_layer):

  cell= Build_RNN_layer(Parametres_layer)
  layer=L.Bidirectional(cell)

  return  layer 

# contruit un réseau feedforward 
def Build_MLP_Character_Dist(Parametres_mlp_CharDist,decoder_combined_context,vocabulary_length):

  output=decoder_combined_context
  for parametre in Parametres_mlp_CharDist:
    output = L.TimeDistributed(L.Dense(units=parametre["units"], activation=parametre["activation"]))(output) # equation (5) of the paper
  output = L.TimeDistributed(L.Dense(vocabulary_length, activation="softmax"))(output) # equation (6) of the paper

  return output 
# contruit une couche RNN selon son type 

def Build_RNN_layer(Parametre_layer):

  if Parametre_layer["type_cell"]== "LSTM" : 
    cell=L.LSTM(units=Parametre_layer["units"],
    dropout=Parametre_layer["dropout"],
    recurrent_dropout=Parametre_layer["recurrent_dropout"],
    return_sequences=True,
    return_state=True,
    stateful=Parametre_layer["stateful"] , unroll=Parametre_layer["unroll"] )   
  elif Parametre_layer["type_cell"]=="CuDNNGRU": 
    cell=CuDNNGRU(units=Parametre_layer["units"],
    dropout=Parametre_layer["dropout"],
    recurrent_dropout=Parametre_layer["recurrent_dropout"],
    return_sequences=True,
    return_state=True,
    stateful=Parametre_layer["stateful"] , unroll=Parametre_layer["unroll"] )
  elif Parametre_layer["type_cell"]=="GRU": 
    cell=L.GRU(units=Parametre_layer["units"],
    return_sequences=True,
    return_state=True,
    stateful=Parametre_layer["stateful"]  )
  else : #by default CuDNNLSTM 
    cell= CuDNNLSTM(units=Parametre_layer["units"],
    
    return_sequences=True,
    return_state=True,
    stateful=Parametre_layer["stateful"]  )

  return cell  

#contruit la couche d'attention selon le type 

def Build_Attention_layer(Parametre_layer , encoder ,decoder ) : 
  
  if Parametre_layer["type_attention"]== "Luong" : 
    # the luong's attention 
    attention = L.dot([decoder[0], encoder], axes=[2, 2])
    attention = L.Activation('softmax')(attention)
    context = L.dot([attention, encoder], axes=[2,1])
    decoder_combined_context = K.concatenate([context, decoder[0]])
  elif Parametre_layer["type_attention"] == "Luong_keras" : 
        # the luong's attention 
    context_vector = L.Attention(use_scale=Parametre_layer["use_scale"],
                                 causal=Parametre_layer["use_self_attention"], 
                                 dropout = Parametre_layer["dropout"] 
                                 )([decoder[0],encoder])
    decoder_combined_context = K.concatenate([context_vector, decoder[0]])
  elif Parametre_layer["type_attention"] == "Bah_keras" : 
        #we are going to use the AditiveAttention = bahd of keras 
    context_vector = L.AdditiveAttention(use_scale=Parametre_layer["use_scale"],
                                 causal=Parametre_layer["use_self_attention"], 
                                 dropout = Parametre_layer["dropout"] )([decoder[0],encoder])
    decoder_combined_context = K.concatenate([context_vector, decoder[0]])
  
  return decoder_combined_context 

# construit le CNN 

def BuildCNN(parametres_CNN,encoder ):

  for para_CNN in parametres_CNN: 
 
    encoder = L.Conv2D(para_CNN["filters"], para_CNN["kernel_size"], activation=para_CNN["activation"], padding='same')(encoder)
    encoder = L.BatchNormalization()(encoder)
    encoder=L.MaxPooling2D(para_CNN["kernel_size"], strides=(2,2), padding='same')(encoder)
 
  return encoder

    


# # Construire le modèle LAS

# In[ ]:


def Listen(input_length,parametres_melspectgtom,parametres_CNN,parametres_BRNN ) : 
    '''
    parametres_melspectgtom : parametre de la couche Melspectrogram si length = 0 donc les audios sont déja mfcc
    parametres_BRNN : a list that contains the parameters of the cells  for each bidectionnel layer 
    then  number_layers is the len of this list parametres_BRNN 
    parametres_CNN: parametres used to build the CNN network after the inputs 
    input_length is the len of the input audios 
    '''
    number_layers = len(parametres_BRNN) 
    encoder_inputs = L.Input(shape=(input_length,))
    
    if parametres_melspectgtom["mfccs"]==False  : 
        #MELSPECTROGRAM Layer 
        encoder_inputs = L.Input(shape=(input_length,))
        encoder = L.Reshape((1, -1))(encoder_inputs)   
        m=Build_MelSpectrogram(parametres_melspectgtom,input_length)
        encoder = m(encoder)
        encoder = Normalization2D( name='mel_stft_norm',str_axis='freq')(encoder)
        # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
        # we would rather have it the other way around for LSTMs (batch_size,timeSteps,melDim,1)
        encoder = L.Permute((2, 1, 3))(encoder)
        encoder = BuildCNN(parametres_CNN,encoder)
        encoder = L.Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim')(encoder)
        
        
    else :
      encoder_inputs = L.Input(shape=(517,13,1))
      encoder = BuildCNN(parametres_CNN,encoder_inputs)
      encoder = L.Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim')(encoder)

      #dans le cas ou nous avons des mfcc  
    inputs=encoder
    
    encoder_state_fbw=None 
    for parametre in parametres_BRNN: 
        print(parametre)
        bltsm_layer = Build_Bidiractionnel_layer(parametre)
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = bltsm_layer(inputs,initial_state=encoder_state_fbw)
        state_h = L.Concatenate()([forward_h, backward_h])
        state_c = L.Concatenate()([forward_c, backward_c])
        encoder_state_fbw = [forward_h, backward_h,forward_c, backward_c]
        inputs = L.Dropout(0.1)(encoder_outputs)
        print("end")
    
    #encoder_state = tuple(encoder_state_fbw * number_layers )
    print("shape of encoder_outupts " , encoder_outputs.shape)
    print("shape of encode_states  " , state_h.shape, state_c.shape)
    
    return  encoder_inputs,encoder_outputs,encoder_state_fbw
    
def AttendAndSpell(encoder,encoder_state_fbw ,output_length,vocabulary_length, parametres_rnns  , parametres_attention 
                   , dim_embedding  ,Parametres_mlp_CharDist   ): 
    
    '''
    encoder= encoder outputs 
    encoder_state_fbw : inputs of the decoder 
    parametres_rnns : parameters used to build the BRNN 
    output_length = length max of sentence ( output shape=150 )
    parametres_attention = parameters  that discribe the attention layer   
         available types : 
         Luong : luong costume layer 
         Luong_keras : luong of keras "attention" 
         Bah_keras : bahdanau attention implemented by keras 
    vocabulary_length: size of the vocabulary (50)
    dim_embedding : dimention of the embedding 
    Parametres_mlp_CharDist: parameters to build the character distribution mlp
    '''
    decoder_inputs = L.Input(shape=(output_length,), name='decoder_inputs')
    decoder_embedding = L.Embedding(vocabulary_length, dim_embedding ,input_length= output_length , name='decoder_embedding')(decoder_inputs)
     #the RNN => si = RNN(si−1, yi−1, ci−1)
    print(decoder_inputs.shape)
    if len(parametres_rnns)!=0 : 
        state_h=K.concatenate([encoder_state_fbw[0], encoder_state_fbw[1]])
        state_c=K.concatenate([encoder_state_fbw[2], encoder_state_fbw[3]])
        initial_sta = [state_h,state_c]
        #first layer with initial state = encoder_states 
        print("jaz 1")
        print(parametres_rnns)
        decoder = Build_RNN_layer(parametres_rnns[0])(decoder_embedding,initial_state = initial_sta)
        print("jaz 2")
        decoder_embedding = decoder 
      # stacked LSTMs with same number of units 
        for param in parametres_rnns[1:]:
          decoder = Build_RNN_layer(param) (decoder_embedding)
          decoder_embedding = decoder 
    else : 
      print("problem  ! ")
    #construct of teh attention 
    decoder_combined_context = Build_Attention_layer(parametres_attention,encoder,decoder)
  #the character distribution layer   
    output=Build_MLP_Character_Dist(Parametres_mlp_CharDist,decoder_combined_context,vocabulary_length)
    '''output = L.TimeDistributed(L.Dense(128, activation="tanh"))(decoder_combined_context) # equation (5) of the paper
    output = L.TimeDistributed(L.Dense(vocabulary_length, activation="softmax"))(output) # equation (6) of the paper
    '''
    return output,decoder_inputs 

def build_model(NAME,input_length,output_length,vocabulary_length, parametre_mel,par_CNN, parametres_brnn  ,dim,
                parametres_dec_rnns , parametres_attention,param_mlp_char_dist ) : 
    
    encoder_inputs,encoder,encoder_state = Listen(input_length,parametre_mel,par_CNN, parametres_brnn ) 
    probabilities,decoder_inputs = AttendAndSpell(encoder,encoder_state ,output_length,vocabulary_length,
                                                  parametres_dec_rnns 
                                                   , parametres_attention 
                                                  , dim  ,param_mlp_char_dist)
    
    model = Model(inputs=[encoder_inputs,decoder_inputs ] , outputs=[probabilities], name=NAME   )
    model.summary()
    with open('/content/drive/My Drive/Colab Notebooks/SpeechRecognition/archis_fitted_models/modelsummary{}.txt'.format(NAME), 'w') as f:
      with redirect_stdout(f):
        model.summary()
    plot_model(model, to_file='/content/drive/My Drive/Colab Notebooks/SpeechRecognition/archis_fitted_models/{}_graph.png'.format(NAME))
    
    return model


# In[ ]:


#les paramétres fixer 

def Parametres (SabOuWidad="widad") :
  #encoder params

  if  SabOuWidad=="widad" : 
    parametre_mel=[{
              "mfccs" :  True , 
              "n_dft": 1024 , 
              "n_hop" :  128, 
             
              "padding" :'same' , 
              "sr" : 22050 , 
            "n_mels" :80  , 
            } ]
    parametres_CNN=[[
                     { "filters" : 128 , 
                      "kernel_size" : (5,1), 
                      "activation" : 'relu'
                         
                     }, 
                      { "filters" : 64 , 
                      "kernel_size" : (5,1), 
                      "activation" : 'relu'
                         
                     }, 
                     {
                         "filters" : 1 , 
                      "kernel_size" : (5,1), 
                      "activation" : 'relu'
                     }
                    ]
                    ]
                    
    
    parametres_brnn=[ [
                      {
                      "type_cell" : "" , 
                      "units": 32,  
                      "stateful" : False  , 
                      "recurrent_dropout" : 0.0, 
                      "dropout"  : 0.1 , 
                      "unroll" : False  
                      } , {
                      "type_cell" : "" , 
                      "units": 32,  
                      "stateful" : False  , 
                      "recurrent_dropout" : 0.0, 
                      "dropout"  : 0.1 , 
                      "unroll" : False  
                      } 
                      
      ]
    ]


    #decoder params 
    parametres_attention=[ 
    {
        "type_attention" : "Bah_keras" , 
        "use_scale": False , 
        "use_self_attention" : True  , 
        "dropout" : 0.0
    }  
     ]
    parametres_dec_rnns = [
                      [
                      {
                      "type_cell" : "" , 
                      "units": 64, 
                      "stateful" : False  , 
                      "unroll" : False   , 
                      "recurrent_dropout" : 0.0, 
                      "dropout"  : 0.1 
                      } , {
                      "type_cell" : "" , 
                      "units": 64, 
                      "stateful" : False  , 
                      "unroll" : False   , 
                      "recurrent_dropout" : 0.0, 
                      "dropout"  : 0.1 
                      }
                      ] ]
    parametres_mlp_CharDist =[
                              [
                               { "units" : 64 , 
                                "activation": "tanh"
                                   
                               },
                               { "units" : 64 , 
                                "activation": "tanh"
                                   
                               }
                              ]
                            ]
    dims_embedding=[200]
    parametres_compil=[
                    {
        "default": True , 
        "name_opt" : "nadam" , 
        "decay" : 0.95, 
        "lr" :  0.05
    } 
    ]
    parametres_fit = [{
        "epochs" : 1000 , 
        "batch_size": 32 , 
        "cross_validation" : True , 
        "type_cross_validation" : "KFold" , 
        "n_splits" : 4 , 
        "shuffle" : True , 
        "random_state": 42 , 
        "initial_epoch" : 0, 
        
    }]

                

   
  return parametre_mel,parametres_CNN,parametres_brnn,dims_embedding,parametres_attention, parametres_dec_rnns,parametres_mlp_CharDist, parametres_compil , parametres_fit   


# # Apprentissage 

# In[ ]:


# cette fonction compile le modéle en précisant l 'optimiseur avec ses paramétres 

def compileModel(model,parametres_compil): 

  if parametres_compil["default"]  == True : 
        opt=parametres_compil["name_opt"] 
  else : 
    if parametres_compil["name_opt"] =="rmsprop" : 
      opt = tf.keras.optimizers.RMSprop(
              learning_rate=parametres_compil["lr"],
              rho=parametres_compil["decay"],
              momentum=parametres_compil["momentum"],
              epsilon=1e-07,
              centered=False   
            )
    elif parametres_compil["name_opt"] =="adam" : 
      
      opt=  tf.keras.optimizers.Adam(
        learning_rate=parametres_compil["lr"],
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False 
      )
    elif parametres_compil["name_opt"] =="adadelta" : 
      opt = tf.keras.optimizers.Adadelta(
          learning_rate=parametres_compil["lr"], rho=parametres_compil["decay"], epsilon=1e-6
        )
    elif parametres_compil["name_opt"] =="adamax" : 
      opt=  tf.keras.optimizers.Adamax(
        learning_rate=parametres_compil["lr"], beta_1=0.9, beta_2=0.999, epsilon=1e-07
        )
    elif parametres_compil["name_opt"] =="nadam" : 
      
      opt= tf.keras.optimizers.Nadam(
          learning_rate= parametres_compil["lr"], beta_1=0.9, beta_2=0.999, epsilon=1e-07     
          )
    elif parametres_compil["name_opt"] =="sgd" : 
     
      opt = tf.keras.optimizers.SGD(
             learning_rate= parametres_compil["lr"], momentum=parametres_compil["lr"], nesterov=False 
          )
    
  model.compile(optimizer=opt, loss=['categorical_crossentropy'], metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy", dtype=None)])

  return model 

# défintion des callbaks : tensorboard , early stopper et reduce lr et le checkpointer 

def Callbacks(NAME): 

  earlystopper = EarlyStopping(monitor='val_loss', patience=10,verbose=1, restore_best_weights=True)
  checkpointer = ModelCheckpoint('./checkpoints_models/model-checkpoint{}.h5'.format(NAME), monitor='val_loss',
                         save_best_only=True, save_weights_only=False,
                         mode='auto', save_freq='epoch', verbose=1)
  callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=0,
                                       verbose=1)
  callback_tensorboard =TensorBoard(log_dir="/content/drive/My Drive/Colab Notebooks/SpeechRecognition/tensorboard_logs/{}".format(NAME),
                                   histogram_freq=1,
                                   write_graph=True, profile_batch=0)
  return  [earlystopper,checkpointer,callback_reduce_lr,callback_tensorboard]

# evaluation du modéle 
def Evaluate_model(model,validation_encoder_input,validation_decoder_input,validation_decoder_output): 
    history = model.evaluate(
    x=[validation_encoder_input, validation_decoder_input],
    y=[validation_decoder_output] ) 
    return history 
#apprentissage et enregistrement du modéle 

def Fit_And_Save_And_Evaluate(model,NAME, parametres_fit, training_encoder_input
                              , training_decoder_input
                              ,training_decoder_output
                              ,validation_encoder_input
                              ,validation_decoder_input, validation_decoder_output): 

  print(training_encoder_input.shape)
  print(training_decoder_input.shape)
  print(training_decoder_output.shape)
  results = model.fit(
      x=[training_encoder_input, training_decoder_input],
      y=[training_decoder_output],shuffle=True,
      validation_data=([validation_encoder_input, validation_decoder_input], [validation_decoder_output]),
      verbose=2,batch_size=parametres_fit["batch_size"], callbacks=Callbacks(NAME)
      , epochs=parametres_fit["epochs"],initial_epoch=0)
  evaluation_history=Evaluate_model(model,validation_encoder_input,validation_decoder_input,validation_decoder_output)
  model.save('/content/drive/My Drive/Colab Notebooks/SpeechRecognition/fitted_moddels/{}_{}.h5'.format(NAME,evaluation_history))
  
  return model , results  , evaluation_history

# apprentissage d'une configuration 
def One_step_main(path_csv,parametre_mel,par_CNN, parametres_brnn,dim,parametres_dec_rnns , 
                  parametres_attention ,param_mlp_char_dist, parametres_compil , parametres_fit): 
  df=pd.read_csv(path_csv,sep=";")
  if parametres_fit["cross_validation"] == False : 
    input_length,output_length,vocabulary_length,training_encoder_input, training_decoder_input,training_decoder_output,validation_encoder_input,validation_decoder_input, validation_decoder_output = Load_resempaled_data(shuffle=parametres_fit["shuffle"] ,split=True,mfccs=parametre_mel["mfccs"] )
    NAME = "Modele_SpeechReco_mfccs{}".format(int(time.time()))  # a unique name for the model 
    print("building the model ", NAME)
    #les parametres    
    model=build_model(NAME,input_length,output_length,vocabulary_length, parametre_mel,par_CNN, parametres_brnn  ,dim,
                  parametres_dec_rnns , parametres_attention,param_mlp_char_dist )
    model =compileModel(model, parametres_compil )
    model , results , evaluation_history = Fit_And_Save_And_Evaluate(model,NAME,parametres_fit
                                                                     ,training_encoder_input
                                                                     , training_decoder_input
                                                                     ,training_decoder_output
                                                                     ,validation_encoder_input
                                                                     ,validation_decoder_input, validation_decoder_output)
    df = Update_csv_params(df,NAME,evaluation_history,parametres_fit,parametres_compil , path_csv)
    
  else  : 
    # ici on execute la cross_validation 
    input_length,output_length,vocabulary_length,training_encoder_input, training_decoder_input,training_decoder_output= Load_resempaled_data(shuffle=parametres_fit["shuffle"] ,split=False,mfccs=parametre_mel["mfccs"] )

    if parametres_fit["type_cross_validation"] == "KFold": 
      kf = KFold(n_splits =  parametres_fit["n_splits"] ,shuffle= parametres_fit["shuffle"] 
                 , random_state=parametres_fit["random_state"])

    elif parametres_fit["type_cross_validation"] == "StratifiedKFold": 
      kf = StratifiedKFold(n_splits =  parametres_fit["n_splits"] ,shuffle= parametres_fit["shuffle"] 
                 , random_state=parametres_fit["random_state"])
    i=0  
    #print("aw ? ", training_encoder_input)
    dataset_split=[]
    for train_index, test_index in kf.split(training_encoder_input):
        print("TRAIN:", train_index, "TEST:", test_index)
        train_index = train_index.astype(int)
        test_index = test_index.astype(int)
        X_encoder_train, X_encoder_test = training_encoder_input[train_index], training_encoder_input[test_index]
        X_decoder_train, X_decoder_test = training_decoder_input[train_index], training_decoder_input[test_index]
        y_train, y_test = training_decoder_output[train_index], training_decoder_output[test_index]
        NAME = "Modele_SpeechReco_{}".format(int(time.time()))  # a unique name for the model 
        print("building the model ", NAME)
        #les parametres    
        model=build_model(NAME,input_length,output_length,vocabulary_length, parametre_mel,par_CNN, parametres_brnn  ,dim,
                      parametres_dec_rnns , parametres_attention,param_mlp_char_dist )
        model = compileModel(model, parametres_compil )
        model , results , evaluation_history = Fit_And_Save_And_Evaluate(model,NAME,parametres_fit
                                                                        ,X_encoder_train
                                                                        , X_decoder_train
                                                                        ,y_train
                                                                        ,X_encoder_test
                                                                        ,X_decoder_test, y_test)
        
        k_fold_set = {
                    'k_fold': i,
                    'train': {'X_1': X_encoder_train, 'X_2': X_decoder_train, 'Y': y_train},
                    'test': {'X_1': X_encoder_test, 'X_2': X_decoder_test, 'Y': y_test}
                    }
        df = Update_csv_params(df,NAME,evaluation_history,parametres_fit,parametres_compil , path_csv)
        dataset_split.append(k_fold_set)
        i = i + 1

# ici on modifie un csv qui contient les résultats avec paramétrage 
  
def Update_csv_params(df , NAME, history , param_fit,param_compil, path_csv ) :
    
    line = {"name_model" : NAME , 
            "val_loss" : history[0], 
            "val_acc" : history[1] 
           }
    line.update(param_fit)
    line.update(param_compil)
    #line=team = dict(param_fit.items() + team_b.items())
    df_line = pd.DataFrame([line]) 
    df_appended = df.append(df_line, ignore_index = True)
    df_appended.to_csv(path_csv,sep=";" )
    
    return df_appended 
    
# Main globale     
def MAIN(path_csv = "/content/drive/My Drive/Colab Notebooks/SpeechRecognition/dict_logs.csv") :

  
  parametre_mel,parametres_CNN,parametres_brnn,dims_embedding,parametres_attention, parametres_dec_rnns,parametres_mlp_CharDist, parametres_compil , parametres_fit  =  Parametres(SabOuWidad="widad")
  
  for par_fit in parametres_fit: 
    for par_compil in parametres_compil  : 
      for par_mel in parametre_mel: 
        for par_CNN in parametres_CNN: 
          for par_brnn_enc in parametres_brnn :
            for dim in  dims_embedding : 
              for par_att in parametres_attention : 
                for para_rnn_dec in parametres_dec_rnns: 
                  for param_mlp_char in parametres_mlp_CharDist :
                    One_step_main(path_csv,par_mel,par_CNN,par_brnn_enc, dim, para_rnn_dec,par_att,param_mlp_char,par_compil,par_fit)
                    print("END of the configuration ")
                    #save results in the dataframe of parametres ! 




MAIN()

