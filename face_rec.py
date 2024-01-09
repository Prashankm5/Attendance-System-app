import numpy as np
import pandas as pd
import cv2
import os

import redis

# insight face
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

# datetime
from datetime import datetime


# Redis Conection
host = 'redis-14599.c267.us-east-1-4.ec2.cloud.redislabs.com'
port = 14599
password = 'sP8eiK1CmTHkGoTzGkxA5GLZcoRXXuhp'
r = redis.Redis(host=host,
               port=port,
               password=password)

# Retrieve Data From Database

def retrive_data(name):
    retrive_dict = r.hgetall(name)
    retrive_series = pd.Series(retrive_dict)
    retrive_series = retrive_series.apply(lambda x: np.frombuffer(x,dtype=np.float32))
    index = retrive_series.index
    index = list(map(lambda x: x.decode(),index))
    retrive_series.index = index
    retrive_df=retrive_series.to_frame().reset_index()
    retrive_df.columns=['name_role', 'facial_features']
    retrive_df[['Name', 'Role']]= retrive_df['name_role'].apply(lambda x: x.split('@')).apply(pd.Series)
    return retrive_df[['Name', 'Role', 'facial_features']]


# Configure FaceAnalysis

faceapp = FaceAnalysis(name='buffalo_sc',
                      root='insightface_model/')
faceapp.prepare(ctx_id=0, det_size=(640,640), det_thresh=0.5)


# ML Search Algorithm
def ml_Search_algorithm(datafram, feature_column, test_vector,name_role=['Name','Role'], thresh=0.5):
    """
    cosine Similarity based search algorithm
    """
    # Step-1 Take a datafram (Collection of data)
    datafram = datafram.copy()

    # Step-2 Index Face embedding from the datfram to convert array
    x_list=datafram[feature_column].tolist()
    x = np.array(x_list)
    
    # Step-3 Calculate cosine Similarity
    y = test_vector.reshape(1,-1)
    
    similar = pairwise.cosine_similarity(x,y)
    similar_arr = np.array(similar)
    datafram['cosine_similarity'] = similar_arr
    

    # Step-4 Fillter the data
    data_filter = datafram.query(f"cosine_similarity>={thresh}")
    data_filter.reset_index(drop=True,inplace=True)

    # Step-5 Get The Person Name
    if len(data_filter)>0:
        argmax = data_filter['cosine_similarity'].argmax()
        name,role = data_filter.loc[argmax][name_role]

    else:
        name = 'Unknown'
        role = "Unknown"
    return name,role

### Real Time Prediction
## We need to save log for every 1 mins

class RealTimePred:
    def __init__(self):
        self.logs = dict(name=[],role=[],current_time=[])
        
    def reset_dict(self):
        self.logs = dict(name=[],role=[],current_time=[])
        
    def saveLogs_redis(self):
        # step-1: create a logs dataframe
        dataframe = pd.DataFrame(self.logs)        
        # step-2: drop the duplicate information (distinct name)
        dataframe.drop_duplicates('name',inplace=True) 
        # step-3: push data to redis database (list)
        # encode the data
        name_list = dataframe['name'].tolist()
        role_list = dataframe['role'].tolist()
        ctime_list = dataframe['current_time'].tolist()
        encoded_data = []
        for name, role, ctime in zip(name_list, role_list, ctime_list):
            if name != 'Unknown':
                concat_string = f"{name}@{role}@{ctime}"
                encoded_data.append(concat_string)
                
        if len(encoded_data) >0:
            r.lpush('attendance:logs',*encoded_data)
        
                    
        self.reset_dict()



    def face_prediction(self, text_img,datafram, feature_column,name_role=['Name','Role'], thresh=0.5):

        # Step0. Find the time
        current_time = str(datetime.now())

        # Step1. Take The Test image and apply to insideface
        result = faceapp.get(text_img,max_num=0)
        new_timg = text_img.copy()

        # Step2. Use for loop to extract each embeddings and pass to ml_search algorithm
        for res in result:
            x1,y1,x2,y2 = res['bbox'].astype(int)
            # print(x1,y1,x2,y2)
            test_emadings = res['embedding']
            
            name,role = ml_Search_algorithm(datafram,feature_column,test_vector=test_emadings, name_role=['Name','Role'], thresh=0.5)
            
            # print(name,role)
            if name == 'Unknown':
                color = (0,0,255)
            else:
                color = (0,255,0)
            cv2.rectangle(new_timg,(x1,y1),(x2,y2),color,1)
            cv2.putText(new_timg,f"{name}",(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.5,color,1)
            cv2.putText(new_timg,current_time,(x1,y2+10),cv2.FONT_HERSHEY_DUPLEX,0.5,color,1)

            # Save info in log dict
            self.logs['name'].append(name)
            self.logs['role'].append(role)
            self.logs['current_time'].append(current_time)
        return new_timg


# Registration Form
class RegistrationForm:
    def __init__(self):
        self.sample = 0
    def reset(self):
        self.sample = 0
        
    def get_embedding(self,frame):
        # get results from insightface model
        results = faceapp.get(frame,max_num=1)
        embeddings = None
        for res in results:
            self.sample += 1
            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),1)
            # put text samples info
            text = f"samples = {self.sample}"
            cv2.putText(frame,text,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,0),2)
            
            # facial features
            embeddings = res['embedding']
            
        return frame, embeddings
    

    def save_data_in_redis_db(self, name, role):
        # validation name
        if name != None:
            if name.strip() != '':
                key = f'{name}@{role}'
            else:
                return 'naem_false'
        else:
            return 'name_false'
        
        if 'face_embeddings.txt' not in os.listdir():
            return 'file_false'

        #Load face_embeddings.txt
        x_array = np.loadtxt('face_embeddings.txt', dtype=np.float32)

        # Convert into proper shape
        recived_sample = int(x_array.size/512)
        x_array = x_array.reshape(recived_sample, 512)
        x_array = np.asarray(x_array)

        # Mean of embeddings
        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes = x_mean.tobytes()

        # save into redis db
        r.hset(name='academy:register', key=key, value=x_mean_bytes)
        os.remove('face_embeddings.txt')
        self.reset()

        return True


