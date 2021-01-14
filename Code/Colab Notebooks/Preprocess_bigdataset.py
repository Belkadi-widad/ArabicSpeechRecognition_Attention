#!/usr/bin/env python
# coding: utf-8

# In[40]:


import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
#from pydub import AudioSegment
from os import listdir
from os.path import isfile, join
import librosa  as rosa 
import math 

import IPython.display as ipd
import soundfile as sf
import pandas as pd 
from tensorflow.keras import backend as K
import pickle
import tensorflow as tf 
import audioread
from sklearn.model_selection import train_test_split 


# In[57]:


def Plot_audio(x, Fs, text=''): 
    
    print('%s Fs = %d, x.shape = %s, x.dtype = %s' % (text, Fs, x.shape, x.dtype))
    plt.figure(figsize=(10, 2))
    plt.plot(x, color='gray')
    plt.xlim([0, x.shape[0]])
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
    ipd.display(ipd.Audio(data=x, rate=Fs))


def get_y_x_after_taksim(top_db=60,path_x=r".\arabic-speech-corpus\arabic-speech-corpus\wav"
                         ,path_y=r".\arabic-speech-corpus\arabic-speech-corpus\orthographic-transcript.txt",max_len=1000000000,seuil=150):
    
    x_train= []
    y_train= []
    dict_index,_,_,dict_file_length=GetInfos([path_y],"DICT")
    j=0
    for key,value in dict_file_length.items():  
        x,sr = rosa.load(path_x+"\{}".format(key))
        '''with audioread.audio_open(path_x+"\{}".format(key)) as f:
                print("duration" , f.duration)''' 
        nonMuteSections =rosa.effects.split(x,top_db)
        expected_partition=value.count("-")+1
        if nonMuteSections.shape[0]==expected_partition:
            #yat9assam
            print(Test_Transcription(text=value))
            for i in range(0,len(nonMuteSections)):  
                s=x[nonMuteSections[i][0]:nonMuteSections[i][1]]
                duration=rosa.get_duration(s,sr)
                if duration>=1.0: 
                    y=value.split("-")[i]
                    if y.count("\"")==0: 
                        y=y+"\""
                    if len(y) <= seuil :
                        x_train.append(x[nonMuteSections[i][0]:nonMuteSections[i][1]])
                        y_train.append(Sentence(y,"DICT",dict_index))
        else : 
            #mayata9ssmch ! 
            duration=rosa.get_duration(x,sr)
            if duration>=1.0 and  len(value) <= seuil :
                x_train.append(x)
                y_train.append(Sentence(value,"DICT",dict_index))
        if j>max_len: 
            break
        j=j+1
        
    return  x_train,y_train

    
def MaxLen_input_output(x_train_total,y_train_total): 
    max_duration=0
    max_len=0
    durations=[]
    les_100=0
    les_200=0
    les_300=0
    for i in range(len(x_train_total)): 
        audio=x_train_total[i]
        len_sentence=len(y_train_total[i])
        if(len_sentence > max_len) :
            max_len=len_sentence
        if (len_sentence < 100): 
            les_100+=1
        elif (len_sentence < 200):
            les_200+=1
        else :
            les_300+=1
        duration = rosa.get_duration(audio)
        durations.append(duration)
        if max_duration < duration :
            max_duration=duration
    print("max duration is ",max_duration)
        
    max_duration=math.ceil(max_duration)
    print("entre 0 et 100" ,les_100)
    print("entre 100 et 200", les_200)
    print("200 ou tala3 ", les_300)
    
    return max_duration, max_len, durations


    
def sort_pad(x_train,y_train,frmt,dict_voc,len_longer_sent,sort,path_indices,max_duration,durations): 
    

    if not isfile(path_indices) :
        indices = [txt[0] for txt in sorted(enumerate(y_train), key=lambda x: len(x[1]))]
        print(path_indices)
        #np.save( path_indices, indices , allow_pickle = True , fix_imports = True )
    else :
        if sort==True : 
            indices=np.load(path_indices)
        else: 
            indices=[]
    #le sort 
    if sort== True : 
        print("les indices " , indices)
        y_train = np.array(y_train)[indices]
        x_train=np.array(x_train)[indices]
        durations=np.array(durations)[indices]
        x_train=x_train.tolist()
    #le pad 
    
    for i in range(0,len(y_train)):
        #le pad de y  ! 
        if  len(y_train[i]) < len_longer_sent :
            if frmt == "ASCI" :
                y_train[i]=np.append( y_train[i],np.full(len_longer_sent- len(y_train[i]) , float(ord("_")),dtype=np.float32 ))
            elif frmt=="OneHot" :
                #y_train[i]=np.append( y_train[i],np.full(len_longer_sent- len(y_train[i]) , np.array(string_vectorizer("_",dict_voc)) ))
                for j in range(len(y_train[i])-1,len_longer_sent-1): 
                     y_train[i].insert(j,string_vectorizer("_",dict_voc)[0])
            elif frmt=="DICT" :
                    y_train[i]=np.append( y_train[i],np.full(len_longer_sent- len(y_train[i]) , float(dict_voc["_"]),dtype=np.float32 ))
        # le pad du x         
        pad_ms=(max_duration-durations[i]) *22050
        print("old duration ", rosa.get_duration(x_train[i]))
        x_train[i]=np.append(x_train[i],np.zeros(int(round(pad_ms))))
        print("new duration ", rosa.get_duration(x_train[i]))
    
    y_train_temp = []
    if frmt=="OneHot":
        for row in y_train:
            desired_row = []
            for col in row:
                for char in col: 
                    desired_row.append(float(char))
            y_train_temp.extend(desired_row)
        y_train_fin = np.array(y_train_temp,dtype=np.float32 )
        y_train_fin.resize((y_train.shape[0],len_longer_sent,len(dict_voc) ))
    else : 
        for row in y_train:
            desired_row = []
            for col in row:
                desired_row.append(float(col))
            y_train_temp.extend(desired_row)
        y_train_fin = np.array(y_train_temp,dtype=np.float32 )
        y_train_fin.resize((len(y_train),len_longer_sent ))

    #y_train =y_train.real.astype(np.float32)
    x_train = np.array(x_train,dtype=np.float32)
    
    #x_train.resize((x_train.shape[0],374850 ))
    
    return indices,x_train,y_train_fin

def splitTrainTest(x_train_total,y_train_total,durations,test_split,shuffle): 
    
    return train_test_split(x_train_total,y_train_total,durations,test_size=test_split,shuffle=shuffle)

def Preprocess_data(paths_x,paths_y,sort,test_split):

    if test_split!= 0.0 : 
        
        if sort==False :     
            
            path_pickle_input_train="/content/drive/My Drive/Colab Notebooks/SpeechRecognition/x_train_big.npy"
            path_pickle_output_train="/content/drive/My Drive/Colab Notebooks/SpeechRecognition/y_train_big.npy"
            path_pickle_input_test="/content/drive/My Drive/Colab Notebooks/SpeechRecognition/x_test_big.npy"
            path_pickle_output_test="/content/drive/My Drive/Colab Notebooks/SpeechRecognition/y_test_big.npy"
        else :
            path_pickle_input_train="x_train_big_sort.npy"
            path_pickle_output_train="y_train_big_sort.npy"
            path_pickle_input_test="x_test_big_sort.npy"
            path_pickle_output_test="y_test_big_sort.npy"
                
        path_indices_train="indices_final_train.npy"
        path_indices_test="indices_final_test.npy"
    else : 
        if sort==False: 
            path_pickle_input_train="x_big.npy"
            path_pickle_output_train="y_big.npy"
        else : 
            path_pickle_input_train="/content/drive/My Drive/Colab Notebooks/SpeechRecognition/x_big_sort.npy"
            path_pickle_output_train="/content/drive/My Drive/Colab Notebooks/SpeechRecognition/y_big_sort.npy"  
        path_indices_big="indices_final_big.npy"
    # on commence par wav , wavtest , omba3d ta3 widad  
    if not isfile(path_pickle_input_train) and  not isfile(path_pickle_output_train) : 
        '''
            1- n9assam les x  et ndir l 'équivalent ta3hom le y !! '
            2-  je construit les indices !! 
            3- sort  des audios et les y ! si sort=True !!  
            3-  pad des audios + pad des y ! 
            
        '''
        # les audios + les y  
        dict_voc=CreateVoc(None)
        
        x_train_total =[]
        y_train_total=[]
        for i in range(0,len(paths_x)) : 
            path_x=paths_x[i]
            path_y=paths_y[i]
            x_train,y_train=get_y_x_after_taksim(60,path_x,path_y)
            x_train_total.extend(x_train)
            y_train_total.extend(y_train)
        print("x and y !! ")
        print(len(x_train_total))
        print(len(y_train_total))
        
            #audios = [f for f in listdir(my_path_x) if isfile(join(my_path_x, f))]
        max_length_input,max_length_output,durations=MaxLen_input_output(x_train_total,y_train_total)
        print("infos!!!")
        print(max_length_input,max_length_output,durations , " ", len(durations))
        
        if test_split != 0.0: 
            
            x_train,x_test,y_train,y_test,durations_train,duration_test = splitTrainTest(x_train_total,y_train_total,durations,test_split,shuffle=not sort)
    
            indices_train,x_train,y_train=sort_pad(x_train,y_train,"DICT",dict_voc,max_length_output,sort,path_indices_train,
                                               max_length_input,durations_train)
            indices_test,x_test,y_test=sort_pad(x_test,y_test,"DICT",dict_voc,max_length_output,sort,path_indices_test,
                                                   max_length_input,duration_test)

            # save the numpy 
            
            np.save( path_pickle_input_train, x_train , allow_pickle = True , fix_imports = True )
            np.save( path_pickle_input_test, x_test , allow_pickle = True , fix_imports = True )
            np.save( path_pickle_output_train, y_train , allow_pickle = True , fix_imports = True )
            np.save( path_pickle_output_test, y_test , allow_pickle = True , fix_imports = True )
            
            return x_train,x_test,y_train,y_test,dict_voc

        else : 
    
            indices_train,x_train,y_train=sort_pad(x_train_total,y_train_total,"DICT",dict_voc,max_length_output,sort,path_indices_big,
                                               max_length_input,durations)
           
            # save the numpy 
            np.save( path_pickle_input_train, x_train , allow_pickle = True , fix_imports = True )
            np.save( path_pickle_output_train, y_train , allow_pickle = True , fix_imports = True )
            
            return x_train,y_train,dict_voc

            
    else : 
        dict_voc=CreateVoc(None)
        if test_split!=0.0: 
            x_train=np.load( path_pickle_input_train )
            x_test=np.load( path_pickle_input_test  )
            y_train=np.load( path_pickle_output_train )
            y_test=np.load( path_pickle_output_test)
            return x_train,x_test,y_train,y_test,dict_voc
        else:
          print("coucou1")
          x_train=np.load( path_pickle_input_train)
          y_train=np.load( path_pickle_output_train)
          print("coucou")
          print(x_train.shape)
          print(y_train.shape)
          return x_train,y_train,dict_voc
            

    return None

def sort_x(x_train,indices):
    
    x_train = np.array(x_train)[indices]
    return x_train
    
def visualize_melSpec(): 
    import librosa 
    melspectrogram = librosa.feature.melspectrogram(
            y=x_train[0], sr=22052 )
    S_dB = librosa.power_to_db(melspectrogram, ref=np.max)
    plt.figure(figsize=(17,6))
    plt.pcolormesh(S_dB)

    plt.title('Spectrogram visualization - librosa')
    plt.ylabel('Frequency')
    plt.xlabel('Time')

    plt.show()

    print('melspectrogram.shape', melspectrogram.shape)
    print(melspectrogram)
    S = librosa.feature.inverse.mel_to_stft(M)
    y = librosa.griffinlim(S)

    Plot_audio(y,22052)

def build_mfccs(x_train): 
    
    mfccs=[]
    for x in x_train: 
        # extract MFCCs
        MFCCs = rosa.feature.mfcc(x, n_mfcc=13)
        mfccs.append(MFCCs.T.tolist())
    
    return np.array(mfccs) 

    
def visualize_mfcc(mfcc): 
    
    # Padding first and second deltas
    delta_mfcc  = rosa.feature.delta(mfcc)
    delta2_mfcc = rosa.feature.delta(mfcc, order=2)

    # We'll show each in its own subplot
    plt.figure(figsize=(12, 6))

    plt.subplot(3,1,1)
    rosa.display.specshow(mfcc)
    plt.ylabel('MFCC')
    plt.colorbar()

    plt.subplot(3,1,2)
    rosa.display.specshow(delta_mfcc)
    plt.ylabel('MFCC-$\Delta$')
    plt.colorbar()

    plt.subplot(3,1,3)
    rosa.display.specshow(delta2_mfcc, sr=sr, x_axis='time')
    plt.ylabel('MFCC-$\Delta^2$')
    plt.colorbar()

    plt.tight_layout()

    # Stacking these 3 tables together into one matrix
    M = np.vstack([mfcc, delta_mfcc, delta2_mfcc])

 


# In[58]:


def Test_Transcription(text="" , trans= True): 
    from lang_trans_master.lang_trans_master.lang_trans.arabic import buckwalter

    #print(buckwalter.trans(u'صِفْرْ - وَاحِدْ - اثْنَانْ - ثَلَاثَة - أَرْبَعَة - خَمْسَة - سِتَّة - سَبْعَة - ثَمَانِيَة - تِسْعَة \"')) 
    
    #print(buckwalter.trans(u'التَّنْشِيط - التَّحْوِيل - الرَّصِيد - التَّسْدِيد - نَعَمْ - لَا - التَّمْوِيل- الْبَيَانَات - الْحِسَاب - إِنْهَاء \"')) 
     
    #print(buckwalter.trans(u'الْجَنَّةَ تَحْتَ أَقْدَامِ الْأُمَّهَات')) 
    if trans : 
        return buckwalter.trans(text)
    #print(buckwalter.untransliterate("Siforo - waAHido - AvonaAno - valaAvap - >arobaEap - xamosap - sit~ap - saboEap - vamaAniyap - tisoEap "))
    else : 
        return buckwalter.untransliterate(text)

def CreateVoc(lines):
    
     
    dict_file="/content/drive/My Drive/Colab Notebooks/SpeechRecognition/dict_chars.pkl"

    if  isfile(dict_file): 
      a_file = open(dict_file, "rb")
      dict_vocab = pickle.load(a_file)

    else : 
        dict_vocab={ "@" : 0 , "_" : 1 , "\"": 2 }
        index=3
        for line in lines : 
            sentence=line[0].split(" \"")[1]
            for c in sentence : 
                if not c in dict_vocab :
                    dict_vocab[c]=index 
                    index= index + 1 
       
            
        a_file = open(dict_file, "wb")
        pickle.dump(dict_vocab, a_file)

    print("dict index  " ,dict_vocab )
   
    return dict_vocab


def All_sentences_test_train(paths):
    #nous donne touts les sentences train + test!!! 
    
    lines=[]
    for path in paths : 
        df=pd.read_csv(path,header=None,sep="", delimiter='\n' ,dtype=str)
        print("lines of path ", path)
        lines_path= df.to_numpy(dtype=str).tolist()
        lines.extend(lines_path)
    return lines 

def Max_length_sentence(lines): 
    
    somme_length=0
    dict_kbar={}
    max_length=0
    min_length=20000
    for line in lines : 
        length=len(Get_Characterrs_Sentence(line))
        if length > max_length : 
            max_length= length
        if length<min_length: 
            min_length=length
        somme_length+=length
        dict_kbar[line[0].split(" \"")[0]]=line[0].split(" \"")[1]
        
    print("min length",min_length)
    return max_length, somme_length/len(lines),dict_kbar

def GetInfos(paths,frmt): 
    
    lines=All_sentences_test_train(paths)
    max_length,moyenne,dict_kbar= Max_length_sentence(lines)
    print("biggest sentence " , max_length)
    if frmt=="DICT":
        dict_index=CreateVoc(lines)
        return dict_index,max_length,moyenne,dict_kbar
    else :  
        return {},max_length,moyenne,dict_kbar
    return None


def Load_data(frmt="DICT",split=True, mfccs=False , chiffres=True ): 
    
    if split==False :
        if shuffle==False : 
            #pour le k fold ! 
            x_train,y_train,dict_voca=Preprocess_data([r".\arabic-speech-corpus\arabic-speech-corpus\wav",
                                                  r".\arabic-speech-corpus\arabic-speech-corpus\test set\wav"],[r".\arabic-speech-corpus\arabic-speech-corpus\orthographic-transcript.txt",r".\arabic-speech-corpus\arabic-speech-corpus\test set\orthographic-transcript.txt"]
                                                ,True,0.0)
                
        else : 
            #not splited 
            x_train,y_train,dict_voca=Preprocess_data([r".\arabic-speech-corpus\arabic-speech-corpus\wav",
                                                r".\arabic-speech-corpus\arabic-speech-corpus\test set\wav"],[r".\arabic-speech-corpus\arabic-speech-corpus\orthographic-transcript.txt",r".\arabic-speech-corpus\arabic-speech-corpus\test set\orthographic-transcript.txt"],
                                                True,0.0)
        if chiffres : 
            x_path="/content/drive/My Drive/Colab Notebooks/SpeechRecognition/x_chiffres.npy"
            y_path= "/content/drive/My Drive/Colab Notebooks/SpeechRecognition/y_chiffres.npy"
            x_chiffres,y_chiffres=Preprocess_chiffres_dataset(x_path,y_path)  
            x_train=np.concatenate((x_train,x_chiffres))
            print(" le shape de x " , x_train.shape)
            
            y_train=np.concatenate((y_train,y_chiffres))
            print(" le shape de y " , y_train.shape)   
       

        return x_train,y_train,dict_voca
    else : 
        if shuffle==False : 
            
          x_train,x_test,y_train,y_test,dict_voca=Preprocess_data([r".\arabic-speech-corpus\arabic-speech-corpus\wav",
                                                                r".\arabic-speech-corpus\arabic-speech-corpus\test set\wav"],[r".\arabic-speech-corpus\arabic-speech-corpus\orthographic-transcript.txt",r".\arabic-speech-corpus\arabic-speech-corpus\test set\orthographic-transcript.txt"]
                                                               ,False,0.30)
        else : 
          x_train,x_test,y_train,y_test,dict_voca=Preprocess_data([r".\arabic-speech-corpus\arabic-speech-corpus\wav",
                                                                r".\arabic-speech-corpus\arabic-speech-corpus\test set\wav"],[r".\arabic-speech-corpus\arabic-speech-corpus\orthographic-transcript.txt",
                                                                r".\arabic-speech-corpus\arabic-speech-corpus\test set\orthographic-transcript.txt"]
                                                               ,False,0.30)
        if mfccs : 
          path_mfcc_test="/content/drive/My Drive/Colab Notebooks/SpeechRecognition/test_3083_mfcss.npy"
          path_mfcc_train="/content/drive/My Drive/Colab Notebooks/SpeechRecognition/train_3083_mfcss.npy"
          if not isfile(path_mfcc_train):
            x_train=build_mfccs(x_train)
            x_test=build_mfccs(x_test)
          else : 
            x_train=np.load(path_mfcc_train)
            x_test=np.load(path_mfcc_test)
        
        return x_train,y_train,x_test,y_test,dict_voca
def getData(mfccs=False , split=True ):
  path_pickle_input_train="/content/drive/My Drive/Colab Notebooks/SpeechRecognition/x_train_big.npy"
  path_pickle_output_train="/content/drive/My Drive/Colab Notebooks/SpeechRecognition/y_train_big.npy"
  path_pickle_input_test="/content/drive/My Drive/Colab Notebooks/SpeechRecognition/x_test_big.npy"
  path_pickle_output_test="/content/drive/My Drive/Colab Notebooks/SpeechRecognition/y_test_big.npy"
  dict_voc = CreateVoc(None)
  y_train=np.load(path_pickle_output_train)
  y_test=np.load(path_pickle_output_test)

  if mfccs : 
    path_mfcc_test="/content/drive/My Drive/Colab Notebooks/SpeechRecognition/test_3083_mfcss.npy"
    path_mfcc_train="/content/drive/My Drive/Colab Notebooks/SpeechRecognition/train_3083_mfcss.npy"
    x_train=np.load(path_mfcc_train)
    x_test=np.load(path_mfcc_test)
  else : 
    x_train= np.load(path_pickle_input_train)
    x_test=np.load(path_pickle_input_test)
  if split == False : 
    x_train=np.concatenate((x_train,x_test))
    y_train=np.concatenate((y_train,y_test))
    return x_train,y_train, dict_voc

  return x_train,y_train,x_test,y_test,dict_voc



def Sentence(line,frmt,dict_voc) : 
    
    char_sentences=Get_Characterrs_Sentence(line)
    
    if frmt == "ASCI" : 
        y = np.array([float(ord(c)) for c in char_sentences],dtype=np.float32)
    elif frmt== "OneHot": 
        list_vector_Onehot=np.array(string_vectorizer(char_sentences,dict_voc))
        y=np.array(list_vector_Onehot)
    else : 
        y = np.array([float(dict_voc[c]) for c in char_sentences],dtype=np.float32)

    return y  

def Get_Characterrs_Sentence(line): 
    
    sentence=line
    if len(line[0].split(" \""))>1:
        sentence=line[0].split(" \"")[1]
    char_sentences=[c for c in sentence]
    char_sentences.insert(0,'@')
    
    return char_sentences


def string_vectorizer(strng, alphabet ):
    
    if mode=="index": 
        vector = [[0.0 if char != letter else 1.0 for char in alphabet] 
                  for letter in strng]
    return vector


    
def Decode_sentence(array_sentence,frmt,dict_voc):
    
    y2=""
    if frmt=="ASCI" : 
        y = np.array([chr(int(asci)) for asci in array_sentence])
        y2=y
    elif frmt=="OneHot" : 
        #pour one hot  
        y = [[get_key(dict_voc,i)  for i in range(len(cara)) if cara[i]==1.0  ] for cara in array_sentence ]
        y2=[get_key(dict_voc,i)  for cara in array_sentence for i in range(len(cara)) if cara[i]==1.0   ]
    elif frmt=="DICT" : 
        y = [get_key(dict_voc,cara) for cara in array_sentence ]
        y2=y

    return y,''.join(y2) 

def Decode_sentences(y_train,frmt,dict_voc): 
    sentences=[]
    for y in y_train :
        sen,sen2= Decode_sentence(y,frmt,dict_voc)
        sentences.append(sen2)
    return sentences 

def get_key(my_dict,val):
    
    for key, value in my_dict.items(): 
         if val == value: 
                return key 
  
    return "key doesn't exist"

'''x_train,y_train,dict_voca= Load_data(split=False,shuffle=False)
print("kamal 1 ")
x_train,y_train,x_test,y_test,dict_voca= Load_data(split=True,shuffle=True)
print("kamal 2 ")
x_train,y_train,dict_voca= Load_data(split=False,shuffle=True)
print("kamal 3 ")
x_train,y_train,x_test,y_test,dict_voca= Load_data(split=True,shuffle=False)
print("kamal 4 ")'''


# In[64]:


def CreateDict_words() :
    list_words = [u'صِفْرْ' , 'وَاحِدْ' , 'اثْنَانْ' , 'ثَلَاثَة' ,'أَرْبَعَة' , 'خَمْسَة' , 'سِتَّة' , 'سَبْعَة' , 'ثَمَانِيَة' ,'تِسْعَة' ,'التَّنْشِيط' , 'التَّحْوِيل' , 'الرَّصِيد' , 'التَّسْدِيد'  ,'نَعَمْ' , 'لَا' , 'التَّمْوِيل', 'الْبَيَانَات'  , 'الْحِسَاب' , 'إِنْهَاء']
    dict_y_arab={}
    dict_y_fr={}
    for i in range(0,len(list_words)): 
        dict_y_arab[i+1] = list_words[i]
        dict_y_fr[i+1] = Test_Transcription(list_words[i],True)+"\""
    print(dict_y_arab[1])
    print(dict_y_fr[1])
    return dict_y_arab,dict_y_fr
    
    
def Preprocess_chiffres_dataset(path_pickle_x,path_pickle_y): 
    
    
    path_wavs = "E:\SII\S2\AA-RDN\projet\datasets\Dataset_30_Sep\Dataset_30_Sep\Audios"
    if not isfile(path_pickle_x) : 
      list_words = [u'صِفْرْ' , 'وَاحِدْ' , 'اثْنَانْ' , 'ثَلَاثَة' ,'أَرْبَعَة' , 'خَمْسَة' , 'سِتَّة' , 'سَبْعَة' , 'ثَمَانِيَة' ,'تِسْعَة' ,'التَّنْشِيط' , 'التَّحْوِيل' , 'الرَّصِيد' , 'التَّسْدِيد'  ,'نَعَمْ' , 'لَا' , 'التَّمْوِيل', 'الْبَيَانَات'  , 'الْحِسَاب' , 'إِنْهَاء']
      dict_y_arab,dict_y_fr= CreateDict_words()
      dic_voc=CreateVoc(None)
      # a new dict 
      index=len(dic_voc)
      for k, v in dict_y_fr.items():
          for c in v : 
              if not c in dic_voc :
                      dic_voc[c]=index 
                      index=index+1
      
      print(dic_voc)
      path_audios = [f for f in listdir(path_wavs) if isfile(join(path_wavs, f))]
      print(path_audios)
        
      x_train= []
      y_train= []
      durations=[]
      for  path_audio in path_audios : 
        x,sr = rosa.load(path_wavs+"\{}".format(path_audio))
        x_train.append(x)
        key=path_audio.split(".")[2]
        y=Sentence(dict_y_fr[int(key)],"DICT",dic_voc)
        y_train.append(y)
        duration = rosa.get_duration(x)
        durations.append(duration)
      _,x_train,y_train=sort_pad(x_train,y_train,"DICT",dic_voc,151,False,"path_indices",12.0,durations)
      print(x_train.shape)
      print(y_train.shape)
      np.save( path_pickle_x, x_train , allow_pickle = True , fix_imports = True )
      np.save( path_pickle_y, y_train , allow_pickle = True , fix_imports = True )
    else : 
      print("coucou2")
      x_train = np.load(path_pickle_x)
      print(x_train.shape)
      y_train = np.load(path_pickle_y)
      print(y_train.shape)
        # faire le pad et ts 
        
    return  x_train,y_train
    
#x_train,y_train=Preprocess_chiffres_dataset()  
    


# In[71]:


#Plot_audio(x_train[120],22050)


# In[73]:



#Decode_sentence(y_train[120], "DICT", dic_voc)


# In[ ]:




