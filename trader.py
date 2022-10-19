#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.backend import in_test_phase
from keras import backend 
from keras.layers import Input, Lambda
from keras.models import Model
from tensorflow.python.ops.numpy_ops import np_config


#from numba import jit, cuda
from scipy.spatial import distance
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession



import statistics
import gc
import datetime
import csv
import ast
import random
import math
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
#import torch
import sys
import os
import time
import re
import threading

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

np_config.enable_numpy_behavior()

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def load_model(dir):
    try:
        loaded_model = tf.keras.models.load_model(dir)
    except:
        loaded_model= False
    return loaded_model

def preprocess(stock_data_dir):
    stock_id_list= set()
    stock_id_list.add(stock_data_dir)
    
    for stock_history_data_dir in stock_id_list:       
        if os.path.exists(stock_history_data_dir):
           
            title, ext= os.path.splitext(os.path.basename(stock_history_data_dir))
            stock_id= title.split('_')[0]
            data_list= get_history_data(stock_history_data_dir)
            
           
            
            # Standardize
    
            # open-high-low-close
            time_column= [i[0] for i in data_list.copy()]            
            open_price_column= [float(i[1]) for i in data_list.copy()]
            highest_price_column= [float(i[2]) for i in data_list.copy()]
            lowest_price_column= [float(i[3]) for i in data_list.copy()]
            close_price_column= [float(i[4]) for i in data_list.copy()]     
            
            
            start_time= time_column[0]
            end_time= time_column[-1]
            delta_day= convert_2_day_delta(start_time, end_time)
      
            delta_price= lowest_price_column[-1]- lowest_price_column[0]
           
            slope= delta_price/ delta_day
            
            
            quartile_slope_index_list, quartile_slope_list= get_quartile_slope(data_list, time_column, lowest_price_column) # Return data index list, slope list
            
            target_quartile_slope_index_list= []
            for quartile_slope_index in range(len(quartile_slope_list)):
                if quartile_slope_list[quartile_slope_index]> slope:
                    target_quartile_slope_index_list.append(quartile_slope_index)
            
            target_quartile_slope= [ quartile_slope_list[i] for i in target_quartile_slope_index_list]
            target_quartile_index_list= [quartile_slope_index_list[i] for i in target_quartile_slope_index_list]
           
            all_phase_round_range= get_all_phase_round(data_list, target_quartile_slope, target_quartile_index_list)
          
            price_delta_column= get_price_delta_column(data_list) # Add price delta column
            for index, content in enumerate(data_list):
                    content.append(price_delta_column[index])
                    
            all_period_data_list, target_data_range_begin= get_all_periods_data_to_previous_data_level_with_different_columns(data_list, all_phase_round_range) 
            
            new_data_for_agent_phase_list= []             
            print ('Number of periods', np.array(all_period_data_list, dtype= object).shape)
            for target_index, target_data_list in enumerate(all_period_data_list):
                   # print ('data shape in period', np.array(target_data_list, dtype= object).shape)
                    
                    sd_level_list, sd_level_tag_column= get_sd_level(target_data_list, price_delta_column) 
                    data_for_agent_delta_list= target_data_list.copy()
            
                    for data_index in range(len(data_for_agent_delta_list)):
                         
                        data_for_agent_delta_list[data_index].append(sd_level_tag_column[data_index]) # Add sd level column
                   
                    
                    data_for_agent_phase_list= data_for_agent_delta_list.copy()       
                         
                    phase_stage_range_list= get_phase_stage_range(data_for_agent_phase_list)
                    minimum_price_column_target_phase_peroid_range= [float(i[2]) for i in data_for_agent_phase_list]
                    open_price_column_target_phase_peroid_range= [float(i[1]) for i in data_for_agent_phase_list]  
                    target_price_column_target_phase_peroid_range= open_price_column_target_phase_peroid_range
                    phase_stage_column= tag_phase_stage(target_price_column_target_phase_peroid_range, phase_stage_range_list)
 
            
                    for data_index in range(len(data_for_agent_phase_list)):
                        data_for_agent_phase_list[data_index].append(phase_stage_column[data_index]) # Add phase stage column
                      
                        #data_for_agent_phase_list[data_index].append(phase_stage_range_list) # Add phase stage column list
                    
                    new_data_for_agent_phase_list+= data_for_agent_phase_list
               
                  #  agent_name= 'agent_phase_'+ 'train_data_'+ stock_id+ '_period_'+ str(target_index) + '_for_proj'+'.csv'
                  #  save_train_data_dir= os.path.join(os.getcwd(), agent_name)
                    
                  #  save_train_data(save_train_data_dir, data_for_agent_phase_list)
            new_data_for_agent_phase_list= new_data_for_agent_phase_list[target_data_range_begin:]
                    
            agent_name= 'agent_phase_'+ 'train_data_'+ stock_id+'_for_proj'+'.csv'
            save_train_data_dir= os.path.join(os.getcwd() , agent_name)
                   
            save_train_data(save_train_data_dir, new_data_for_agent_phase_list) 
   

            
def init_training(agent_name, stock_id, refresh_input_data, session_list, total_data_list, num_of_layers, model, memory_state, carry_state, save_model_dir, checkpoint_dir, train_data_dir):
    init_train= True
    refresh_input_data_in_training= False
    init_next_layer= False
    save_model_dir=  save_model_dir
    checkpoint_dir= checkpoint_dir 

    state= (memory_state, carry_state)
    layer_count= 0
    session_index= 1
    num_of_layers= num_of_layers
    train_times_count= 0
    callback_stop_count= 0
    callback_stop_val= 5
    best_result_count_2_callback_stop_val= 5
    predict_data_list= None

    model_list= []   
    pred_result_list= []
    
    train_data_list= session_list[session_index][0]
    validation_data_list= session_list[session_index][1]
        
    train_input_data_list, train_input_data_index_list, validation_input_data_list, validation_data_input_index_list, output_train_data_index_list, output_val_data_index_list, _, _= get_input_data_for_other_input(agent_name, train_data_list, validation_data_list, total_data_list, predict_data_list) # Input data
    output_value_list= get_expected_output_data(agent_name, output_train_data_index_list)  # Output expeceted data    
        
   
    save_train_input_data_dir= os.path.join(save_model_dir, 'train_input_data')
    save_validation_input_data_dir= os.path.join(save_model_dir, 'validation_input_data')

    
    if not os.path.exists(save_train_input_data_dir+ '.npy') or refresh_input_data:
       
         
        np.save(save_train_input_data_dir, np.array(train_input_data_list))

       # save_preprocesss_input_data(save_train_input_data_dir, new_train_input_data_list)
    
     
     
        np.save(save_validation_input_data_dir, np.array(validation_input_data_list))
       # save_preprocesss_input_data(save_validation_input_data_dir, new_val_input_data_list)
     
  
    new_train_input_data_list= get_preprocesss_input_data(save_train_input_data_dir)
    new_validation_input_data_list= get_preprocesss_input_data(save_validation_input_data_dir)

    if model:
        model_input = model.input

        if not model_input[0, :, :].shape== np.array(new_train_input_data_list)[0, :, :].shape:
            model= False   
        else:   

            current_result= get_current_result(session_list, session_index, total_data_list, predict_data_list, agent_name, save_model_dir)
            pred_result_list.append(current_result)


    while init_train:
        gc.enable()
        gc.collect()
        print ('pred_result_list', pred_result_list)
        print ('length of train data', len(train_data_list))
        print ('length of validation data', len(validation_data_list))
        

        # Input data

        input_data_list_arr= new_train_input_data_list
        output_value_list_arr= np.array(output_value_list)  # Output data
        
        
        batch_size, time_steps, features= input_data_list_arr.shape
        print ('batch', batch_size, 'time_steps', time_steps, 'feature', features)

        if refresh_input_data_in_training:
            session_list, total_data_list= get_cross_train_rule_data_and_validation_data(train_data_dir)
            train_data_list= session_list[session_index][0]
            validation_data_list= session_list[session_index][1]
        
            train_input_data_list, train_input_data_index_list, validation_input_data_list, validation_data_input_index_list, output_train_data_index_list, output_val_data_index_list, _, _= get_input_data_for_other_input(agent_name, train_data_list, validation_data_list, total_data_list, predict_data_list) # Input data
            output_value_list= get_expected_output_data(agent_name, output_train_data_index_list)  # Output expeceted data    
        
   
            save_train_input_data_dir= os.path.join(save_model_dir, 'train_input_data')
            save_validation_input_data_dir= os.path.join(save_model_dir, 'validation_input_data')
         
            np.save(save_train_input_data_dir, np.array(train_input_data_list))    
            np.save(save_validation_input_data_dir, np.array(validation_input_data_list))      
  
            new_train_input_data_list= get_preprocesss_input_data(save_train_input_data_dir)
            new_validation_input_data_list= get_preprocesss_input_data(save_validation_input_data_dir)
            refresh_input_data_in_training= False



       # if model:
       #     model= dropout_non_connect(model)
    

        if not model:      
            #####  Build model  #####   
            input_layer= on_input_layer(input_data_list_arr, batch_size, time_steps, features)

            layer= input_layer
            init_next_layer= True
            while init_next_layer:
                if layer_count== num_of_layers:
                    break
                
                new_layer, memory_state, carry_state= on_next_layer(layer, layer_count, num_of_layers, state)    

                state= (memory_state, carry_state)
                layer= new_layer
                layer_count+= 1
                
                
            
            output_layer= on_output_layer(output_value_list_arr, layer, time_steps, features) 

            model = tf.keras.Model(inputs= input_layer, outputs= output_layer, name= agent_name+ '_'+ stock_id) # Multi
   


        #### Batch  training
        tf.keras.backend.clear_session()
        history= train_model(input_data_list_arr, output_value_list_arr, model, checkpoint_dir, batch_size)
      
        
        train_times_count+= 1


        val_accuracy_list= history.history['val_accuracy']
        accuracy_list= history.history['accuracy']

        init_pred= False
        if train_times_count% 5== 0 or init_pred:
            if max(np.array(accuracy_list))>=0.9:
                tf.keras.backend.clear_session()
                accuracy= init_predict_on_training(model, agent_name, new_validation_input_data_list, validation_data_input_index_list, validation_data_list, output_val_data_index_list) 
                gc.enable()
                gc.collect()
                print ('Acc :', accuracy)
                pred_result_list.append(accuracy)
                best_result= max(pred_result_list)
                refresh_input_data_in_training= True
                if len(pred_result_list)> 1 and accuracy== best_result:
                    save_model(model, save_model_dir)
                    print ('Save the best result :', accuracy)
                    callback_stop_count= 0

                    unique, counts = np.unique(pred_result_list, return_counts=True)    # Stop training when best results are same.
                    best_result_count_2_callback_stop_count= int(dict(zip(unique, counts))[accuracy])
                    if best_result_count_2_callback_stop_count== best_result_count_2_callback_stop_val:
                        if max(pred_result_list)< 0.6: # Low acc list
                            low_acc_list.append([stock_id, max(pred_result_list)])
                        init_train= False                  
                        break


                callback_stop_count+= 1
            
            if callback_stop_count== callback_stop_val: # Stop training when no more best result
                if max(pred_result_list)< 0.6: # Low acc list
                    low_acc_list.append([stock_id, max(pred_result_list)])
                init_train= False
                break

        if train_times_count% 20== 0:
            if not max(np.array(accuracy_list))>=0.9:
                save_model(model, save_model_dir)
          
      
    return model, (memory_state, carry_state)

def get_reference_data(dir, target_data_begin, target_data_end, target_data_begin_index, target_data_end_index):
    reference_data_list= []
    with open(dir, 'r', newline='', encoding='utf-8-sig') as csvfile:                       
        reader = csv.reader(csvfile, delimiter=',')
        data_list= [i for i in reader]
        for row in data_list[1:]:
            
            for col_index, col in enumerate(row):
                if col_index!= 0:
                    if col== 'null':
                        continue
                    col= float(col)
     
            reference_data_list.append(row)
        
        for reference_data_index, reference_data in enumerate(reference_data_list):
            if 'null' in reference_data:
                continue
                
            if [float(i) for i in target_data_begin]==[float(i) for i in reference_data[1: 5]]:
               
                target_data_begin_index= reference_data_index- 10
            if  [float(i) for i in target_data_end]==[float(i) for i in reference_data[1: 5]]:
               
                target_data_end_index= reference_data_index
                
    return reference_data_list, target_data_begin_index, target_data_end_index

def get_history_data(path):
    history_data_list= []
    reference_data_list= []
    target_data_list= []
    reference_data_dir_0= os.path.join(os.getcwd(), '0050_reference_data.csv')
    reference_data_dir_1= os.path.join(os.getcwd(), '2330_reference_data.csv')
    stock_id= None
    
    with open(path, 'r', newline='', encoding='utf-8-sig') as csvfile:                       
        reader = csv.reader(csvfile, delimiter=',')
        
        for row in reader:
            for col_index, col in enumerate(row):
                if col_index!= 0:
                    if col== 'null':
                        continue
                    
                    col= float(col)
                
            history_data_list.append(row)
            
    target_data_begin= history_data_list[0]
    target_data_end= history_data_list[-1]   
    target_data_begin_index= None
    target_data_end_index= None
    
    reference_data_dir= reference_data_dir_0
    reference_data_list, target_data_begin_index, target_data_end_index= get_reference_data(reference_data_dir, target_data_begin, target_data_end, target_data_begin_index, target_data_end_index)
    
    
    if target_data_begin_index== None and target_data_end_index== None:
        reference_data_dir= reference_data_dir_1
        reference_data_list, target_data_begin_index, target_data_end_index= get_reference_data(reference_data_dir, target_data_begin, target_data_end, target_data_begin_index, target_data_end_index)
       
            
        
    if target_data_begin_index!= None and target_data_end_index!= None:
        target_data_list =  reference_data_list[target_data_begin_index: target_data_end_index]
    
        
    
    return target_data_list






def get_cross_train_rule_data_and_validation_data(dir):
    total_data_list= []
 
    session_list= []
    with open(dir, 'r+', newline='') as csvfile:    
        reader = csv.reader(csvfile, delimiter=',')
        total_data_list= [i for i in reader]
        num_of_total_data= len(total_data_list)
        
        num_of_val_data= int(num_of_total_data* 1/ 10)   # val rate= 0.1
        num_of_train_data= int(num_of_total_data- num_of_val_data)
        session_num= 2    # cross validation
        bottom_boundary= 15

        for session in range(session_num):
            total_data_list_copy= total_data_list[bottom_boundary:].copy()
            validation_data_list= []  
      
            for num in range(num_of_val_data):
                    rand_sel= random.choice(total_data_list_copy)
                    #total_data_list_copy.remove(rand_sel)  

                    validation_data_list.append(rand_sel)
            train_rule_data_list= total_data_list_copy
            session_list.append([train_rule_data_list, validation_data_list])
    
    return session_list, total_data_list


def save_train_and_val_data(dir, data):
    with open(dir, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

def get_train_and_val_data(dir):
    with open(dir, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data_list= []
        for sub_content in reader:
            data_list.append(sub_content)
      
    return data_list

def get_predict_next_day_data(dir):
    with open(dir, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data_list= []
        for sub_content in reader:
            data_list.append(sub_content)
      
    return data_list

def inpu_data_feature_to_binary(data_list):
    for data in data_list:
         if data_list< 0:
            data_list= 0
         else:
            data_list= 1
    return data_list


def get_input_data_for_other_input(agent_name, train_data_list, validation_data_list, total_data_list, predict_data_list):
    train_input_data_list= [] 
    train_input_data_index_list= []
    validation_input_data_list= []
    validation_input_data_index_list= []
    predict_next_day_input_data_list= []
    predict_next_day_input_data_index_list= []
    output_train_data_index_list= []  # Output data index = [stock id, 日期 , specific agent output]
    output_val_data_index_list= []
    plus_representation_coding= 1
    substraction_representation_coding= 0
    total_date_data= [i[0] for i in total_data_list]
    data_range= 9
    stock_id= '0050'
    for train_data in train_data_list[data_range:]:
        if agent_name== 'agent_phase':               # Agent phase train data
            target_data_range= get_previous_data_range(train_data, data_range, total_data_list, total_date_data) # Use previous days data to predict next day
           # target_data_range= get_previous_data_range_use_data_len(train_data, data_range, total_data_list)
          
            data_in_day_level= []
            for target_data in target_data_range:
                if len(target_data)< 10:
                     continue
                
                oepn_price= target_data[8]     # Open price level              
                oepn_price_list= ast.literal_eval(oepn_price)

                
                highest_price= target_data[9] # Highest price level             
                highest_price_list= ast.literal_eval(highest_price)


                lowest_price= target_data[10]            # Lowest price level              
                lowest_price_list= ast.literal_eval(lowest_price)
                   
                close_price= target_data[11]           # Close price level           
                close_price_list= ast.literal_eval(close_price)

                

                delta= [float(target_data[7])]  # Delta level

               # oepn_price_list= inpu_data_feature_to_binary(oepn_price_list)
               # highest_price_list= inpu_data_feature_to_binary(highest_price_list)
               # lowest_price_list= inpu_data_feature_to_binary(lowest_price_list)
               # close_price_list= inpu_data_feature_to_binary(close_price_list)
               

                #feature_list= [0]+ oepn_price_list+ [1]+ highest_price_list+ [2]+ lowest_price_list+ [3]+ close_price_list      
                feature_list= oepn_price_list+ highest_price_list+ lowest_price_list+ close_price_list
                feature_list= [1, 0, 0 ,0 ]+ oepn_price_list+ [0, 1, 0 ,0 ]+ highest_price_list+ [0, 0, 1 ,0 ]+ lowest_price_list+ [0, 0, 0 ,1 ]+ close_price_list

               # feature_list= [0]+ oepn_price_list#+ [1]+ close_price_list 
                data_in_day_level.append(feature_list)
            train_input_data_list.append(data_in_day_level)
            train_input_data_index_list.append([stock_id, train_data[0]])
            output_train_data_index_list.append([stock_id, train_data[0], train_data[14]])

    
    for validation_data in validation_data_list:
        if agent_name== 'agent_phase':   # Agent phase validation data
            target_data_range= get_previous_data_range(validation_data, data_range, total_data_list, total_date_data) # Use previous days data to predict next day
          #  target_data_range= get_previous_data_range_use_data_len(train_data, data_range, total_data_list)
            data_in_day_level= []
            for target_data in target_data_range:
                if len(target_data)< 10:
                     continue

                oepn_price= target_data[8]     # Open price level              
                oepn_price_list= ast.literal_eval(oepn_price)

                
                highest_price= target_data[9] # Highest price level             
                highest_price_list= ast.literal_eval(highest_price)


                lowest_price= target_data[10]            # Lowest price level              
                lowest_price_list= ast.literal_eval(lowest_price)
                   
                close_price= target_data[11]           # Close price level           
                close_price_list= ast.literal_eval(close_price)



                delta= [float(target_data[7])]  # Delta level

               # oepn_price_list= inpu_data_feature_to_binary(oepn_price_list)
               # highest_price_list= inpu_data_feature_to_binary(highest_price_list)
               # lowest_price_list= inpu_data_feature_to_binary(lowest_price_list)
               # close_price_list= inpu_data_feature_to_binary(close_price_list)
                           
               # feature_list= [0]+ oepn_price_list+ [1]+ highest_price_list+ [2]+ lowest_price_list+ [3]+ close_price_list   
              #  feature_list= oepn_price_list+ highest_price_list+ lowest_price_list+ close_price_list      
                feature_list= [1, 0, 0 ,0 ]+ oepn_price_list+ [0, 1, 0 ,0 ]+ highest_price_list+ [0, 0, 1 ,0 ]+ lowest_price_list+ [0, 0, 0 ,1 ]+ close_price_list
              #  feature_list= [0]+ oepn_price_list#+ [1]+ close_price_list 
                data_in_day_level.append(feature_list)
            validation_input_data_list.append(data_in_day_level)
            validation_input_data_index_list.append([stock_id, validation_data[0]])
            output_val_data_index_list.append([stock_id, validation_data[0], validation_data[14]]) # Output index = [stock id, 日期 , phase stage]

    
    if predict_data_list!= None:
    
        total_date_data= [i[0] for i in predict_data_list]
        predict_next_day= predict_data_list[-1]  # The validation data 
        predict_next_date_data_range= get_previous_data_range(predict_next_day, data_range, predict_data_list, total_date_data)

        data_in_day_level= []
        for target_data in predict_next_date_data_range:
                if len(target_data)< 10:
                     continue
                oepn_price= target_data[8]     # Open price level              
                oepn_price_list= ast.literal_eval(oepn_price)

                
                highest_price= target_data[9] # Highest price level             
                highest_price_list= ast.literal_eval(highest_price)


                lowest_price= target_data[10]            # Lowest price level              
                lowest_price_list= ast.literal_eval(lowest_price)
                   
                close_price= target_data[11]           # Close price level           
                close_price_list= ast.literal_eval(close_price)



                delta= [float(target_data[7])]  # Delta level

               # oepn_price_list= inpu_data_feature_to_binary(oepn_price_list)
               # highest_price_list= inpu_data_feature_to_binary(highest_price_list)
               # lowest_price_list= inpu_data_feature_to_binary(lowest_price_list)
               # close_price_list= inpu_data_feature_to_binary(close_price_list)
                           
               # feature_list= [0]+ oepn_price_list+ [1]+ highest_price_list+ [2]+ lowest_price_list+ [3]+ close_price_list   
              #  feature_list= oepn_price_list+ highest_price_list+ lowest_price_list+ close_price_list      
                feature_list= [1, 0, 0 ,0 ]+ oepn_price_list+ [0, 1, 0 ,0 ]+ highest_price_list+ [0, 0, 1 ,0 ]+ lowest_price_list+ [0, 0, 0 ,1 ]+ close_price_list
              #  feature_list= [0]+ oepn_price_list#+ [1]+ close_price_list 
                data_in_day_level.append(feature_list)

        predict_next_day_input_data_list.append(data_in_day_level)
        predict_next_day_input_data_index_list.append([stock_id, train_data_list[-1][0]])
    return train_input_data_list, train_input_data_index_list, validation_input_data_list, validation_input_data_index_list, output_train_data_index_list, output_val_data_index_list, predict_next_day_input_data_list, predict_next_day_input_data_index_list
    

def get_predict_previous_data_range(target_predict_data, data_range, total_data_list, total_date_data):
    month_eng_2_index_list= ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    current_data_date= target_predict_data
   
    current_y, current_m, current_d= int(current_data_date[: -4])+1911, int(current_data_date[-4: -2]), int(current_data_date[-2:])  # Get target date range
    
    target_data_range= []
    date_count= 1
    while len(target_data_range)< data_range:
                current_date_2_sec= datetime.datetime(current_y, current_m, current_d).timestamp()
                previous_date_sce= int(current_date_2_sec- 3600* 24* date_count)
                previous_date= time.ctime(previous_date_sce)

                previous_date_y, previous_date_m, previous_date_d= int(previous_date[-4:])- 1911, int(month_eng_2_index_list.index(previous_date[4:7]))+ 1, int(previous_date[8:11])
                if previous_date_m< 10:
                    previous_date_m= str(0)+ str(previous_date_m)
                else:
                    previous_date_m= str(previous_date_m)   
                if previous_date_d< 10:
                    previous_date_d= str(0)+ str(previous_date_d)
                else:
                    previous_date_d= str(previous_date_d)              
                target_previous_date= int(str(previous_date_y)+ previous_date_m+ str(previous_date_d))
                if target_previous_date in total_date_data:
                    
                    target_previous_data_index= total_date_data.index(target_previous_date)
                    target_previous_data= total_data_list[target_previous_data_index]
                    target_data_range.append(target_previous_data)
                  
                date_count+= 1

    return target_data_range


def get_previous_data_range(target_data, data_range, total_data_list, total_date_data):
    month_eng_2_index_list= ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    current_data_date= target_data[0]
   
    current_y, current_m, current_d= int(current_data_date.split('-')[0]), int(current_data_date.split('-')[1]), int(current_data_date.split('-')[2])  # Get target date range
    
    target_data_range= []
    date_count= 1
 
    while len(target_data_range)< data_range:
                current_date_2_sec= datetime.datetime(current_y, current_m, current_d).timestamp()
                previous_date_sce= int(current_date_2_sec- 3600* 24* date_count)
                previous_date= time.ctime(previous_date_sce)
               
                previous_date_y, previous_date_m, previous_date_d= int(previous_date[-4:]), int(month_eng_2_index_list.index(previous_date[4:7]))+ 1, int(previous_date[8:11])
              
                if previous_date_m< 10:
                    previous_date_m= str(0)+ str(previous_date_m)
                else:
                    previous_date_m= str(previous_date_m)   
                if previous_date_d< 10:
                    previous_date_d= str(0)+ str(previous_date_d)
                else:
                    previous_date_d= str(previous_date_d)
                                  
                target_previous_date= str(previous_date_y)+'-'+ previous_date_m+'-'+ str(previous_date_d)
                if target_previous_date in total_date_data:
                    
                    target_previous_data_index= total_date_data.index(target_previous_date)
                    target_previous_data= total_data_list[target_previous_data_index]
                    target_data_range.append(target_previous_data)
                  
                date_count+= 1

    return target_data_range



def get_expected_output_data(agent_name, target_index_data_list):
    # Delta: 20 level
    # Phase: 8 stage
    
    all_output_val_list= []
    target_data_date_list= [i[1] for i in target_index_data_list]
    target_data_value_list= [i[2] for i in target_index_data_list]


  
    if agent_name== 'agent_delta':
        output_val_list = list(np.zeros(20))
    else:
        output_val_list= list(np.zeros(8))
   
    for target_data_value in target_data_value_list:
   
        new_output_val_list= output_val_list.copy()
        new_output_val_list[int(target_data_value)]= 1
        all_output_val_list.append(new_output_val_list)

        
    return all_output_val_list
        

def get_current_result(session_list, session_index, total_data_list, predict_data_list, agent_name, save_model_dir):

    save_validation_input_data_dir= os.path.join(save_model_dir, 'validation_input_data')
    new_validation_input_data_list= get_preprocesss_input_data(save_validation_input_data_dir)

    train_data_list= session_list[session_index][0]
    validation_data_list= session_list[session_index][1]
        
    train_input_data_list, train_input_data_index_list, validation_input_data_list, validation_data_input_index_list, output_train_data_index_list, output_vali_data_index_list, _, _= get_input_data_for_other_input(agent_name, train_data_list, validation_data_list, total_data_list, predict_data_list)

    model= load_model(save_model_dir)
    acc= init_predict_on_training(model, agent_name, new_validation_input_data_list, validation_data_input_index_list, validation_data_list, output_vali_data_index_list)

    
    return acc

def dropout_non_connect(model):
    target_layer_id_list= [1, 2, 3, 4, 5, 6]
    for layer_id in target_layer_id_list:

                target_lstm_layer_weights = model.layers[layer_id].get_weights()[0]
                target_lstm_layer_weights_shape= target_lstm_layer_weights.shape
                target_lstm_layer_weights_size= target_lstm_layer_weights.size
                new_target_lstm_layer_weights= target_lstm_layer_weights.reshape(target_lstm_layer_weights_size)
                
                for weight in new_target_lstm_layer_weights:
                    if weight< 0:
                        weight= 0
                  
                new_target_lstm_layer_weights= new_target_lstm_layer_weights.reshape(target_lstm_layer_weights_shape)
                new_param=  model.layers[layer_id].get_weights()
                new_param[0]= new_target_lstm_layer_weights           
                model.layers[layer_id].set_weights(new_param)
                
    return model
    
  

     

def on_preprocessing_layer_multi_hot(input_data_list):
    layer= tf.keras.layers.CategoryEncoding(num_tokens= 60, output_mode="multi_hot")
    new_input_data= layer(input_data_list).numpy()
    #new_input_data_list= tf.keras.layers.Embedding( # Input_dim: Integer. i.e. maximum integer index + 1.
    #  1000,
    #  len(input_data_list),
    #  embeddings_initializer="uniform",
    #  embeddings_regularizer=None,
    #  activity_regularizer=None,
    #  embeddings_constraint=None,
    #  mask_zero=False,
    #  input_length=None,
    #)(input_data_list)
    return new_input_data



def on_input_layer(input_data_list, batch_size, time_steps, features):
   # input= [samples, time_steps, features] [Number of datas, The length of each data, Each element of the data is a vector of n features]
   # Samples - Number of datas
   # Time steps -   The length of each data
   # Features - Each element of the data is a vector of n features
    input_layer = tf.keras.Input(shape=(time_steps, features), name='input_layer') # shape=(32,) indicates that the expected input will be batches of 32-dimensional vectors.

    return input_layer
  
def on_next_layer(layer, layer_count, num_of_layers, state):
    if state[0]== None:
        state= None
    if layer_count== num_of_layers-1:
        dropout= 0
    else:
        dropout= 0
    number_of_units= 128 

    _, time_steps, features= layer.shape




    
    new_layer, memory_state, carry_state= tf.keras.layers.LSTM(units = number_of_units, input_shape= (time_steps, features), name= 'hidden_layer_'+ str(layer_count), return_sequences=True, stateful= False, return_state= True
                                  , time_major= False, activation="tanh", recurrent_activation="sigmoid", unit_forget_bias=True, 
                                  kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", dropout= dropout, 
                                  use_bias=True)(layer, state)

    
   # if not layer_count== num_of_layers- 2:
   #     new_layer = new_layer[0, :, :]
   #     new_layer= tf.keras.layers.RepeatVector(time_steps)(new_layer)
        

    if layer_count== num_of_layers-1:
        new_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(features))(new_layer) # TimeDistributed applies the same instance of previous layer to each of the timestamps, the same set of weights are used at each timestamp.

    return new_layer, memory_state, carry_state




def on_output_layer(output_value_list, layer, time_steps, features):
    outputs= tf.keras.layers.Flatten(data_format=None)(layer)
    outputs= tf.keras.layers.Dropout(0.2) (outputs, training=True) # 0.2
 
    outputs= tf.keras.layers.Dense(units = len(output_value_list[0]), activation='relu', name='output_dense_layer_0', use_bias=True)(outputs)
    outputs= tf.keras.layers.Dense(units = len(output_value_list[0]), activation='sigmoid', name='output_dense_layer_1', use_bias=True)(outputs)
    print ('outputs shape', outputs.shape)

    return outputs

def train_model(input_data_list, data_expected_output, model, checkpoint_dir, batch_size):
    optimizer_Adam= tf.keras.optimizers.Adam(learning_rate= 1e-3,
                                              beta_1=0.9,
                                              beta_2=0.999,
                                              epsilon=1e-07,
    )

    rms_prop= tf.keras.optimizers.RMSprop(
        learning_rate=1e-3,
        rho=0.9,
        momentum=0.0,
        epsilon=1e-07,
        centered=False,
        name="RMSprop",
    )

    ada_delta= tf.keras.optimizers.Adadelta(
        learning_rate=1e-3, rho=0.95, epsilon=1e-07, name="Adadelta"
    )

    sgd= tf.keras.optimizers.SGD(learning_rate=1e-3)

    binary_crossentropy= tf.keras.losses.BinaryCrossentropy(
                from_logits= True,
                label_smoothing=0.0,
                axis=-1,
                reduction="auto",
                name="binary_crossentropy",
    )

    categorical_crossentropy= tf.keras.losses.CategoricalCrossentropy(
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
      #  reduction="auto",
        name="categorical_crossentropy",
    ) 


    callbacks = [
        tf.keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss", # val_loss
        # "No longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "No longer improving" being further defined as "for at least 2 epochs"
        patience=2,
        verbose=1,
        mode= 'auto'
        ),
        tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir,
        monitor='val_accuracy',
        mode='auto',
        save_weights_only= True,
        save_best_only=True
        )    
    ]




    model.compile(#optimizer= sgd,
              optimizer= optimizer_Adam, 
              # Loss function to minimize
              loss= tf.keras.losses.CategoricalCrossentropy(),
              # List of metrics to monitor    
              metrics=['accuracy', 'mse']
            )

    print ('input_data_list shape', input_data_list.shape)
    print ('data_expected_output shape', data_expected_output.shape)
  
 
   # print('# Fit model on training data')
   # model.summary()
    batch_size= 32
    number_of_epochs= 100

    history = model.fit(input_data_list, 
            data_expected_output, 
        batch_size= batch_size, 
        epochs= number_of_epochs,
        validation_split=0.2,
        callbacks= callbacks,
                )
  
   # plt.plot(history.history['accuracy'])
   # plt.plot(history.history['val_accuracy'])
   # plt.title('Model accuracy')
   # plt.ylabel('Accuracy')
   # plt.xlabel('Epoch')
   # plt.legend(['Train', 'Test'], loc='upper left')
   # plt.show()

 # results = model.evaluate(validation_input_data_list, validation_expected_output, batch_size=128)
 # print("test loss, test acc:", results)
    return history



def save_model(Model, dir):
    Model.save(
        dir,
      overwrite=True,
      include_optimizer=True,
      save_format=None,
      signatures=None,
      options=None,
      save_traces=True,
    )

def load_model(dir):
    
    loaded_model = tf.keras.models.load_model(dir)
   
    return loaded_model


def model_predict(predict_data, model):
    tf.keras.backend.clear_session()
    prediction= model.predict(
        predict_data,
        batch_size=None,
        verbose="auto",
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
      )
    return prediction



def init_predict_on_training(model, agent_name, new_predict_input_data_list, predict_input_data_index, predict_data_list, output_data_index_list): 
    
    predict_input_data_list_arr= new_predict_input_data_list
    expected_output_value_list= get_expected_output_data(agent_name, output_data_index_list)
  
    tp_count= 0
    for predict_data, predict_data_index, expected_output_value_index in zip(predict_input_data_list_arr, predict_input_data_index, expected_output_value_list):
        detail= None
        SE= 0
        time_steps, features= predict_data.shape
        new_predict_data= predict_data.reshape(1, time_steps, features)
        prediction= model_predict(new_predict_data, model)
 
        max_val= np.max(prediction)
        test, value_index= np.where(prediction== max_val)
        if len(value_index)!= 1:
            print (predict_data_index, "can't predict data")
            continue
        taget_expected_output_value_index= expected_output_value_index.index(1)
        if value_index== taget_expected_output_value_index:
            tp_count+= 1
    accuracy=  tp_count/ len(predict_input_data_list_arr)      
    # print ('predict_score', predict_score )
    print ('Acuracy', accuracy)
    
    return accuracy




def init_predict(model, predict_next_day_data_list, predict_next_day_index_data_list): 

    predict_next_day_data_list_arr= np.array(predict_next_day_data_list)
    tp_count= 0
    
    for predict_data_index, predict_data in zip(predict_next_day_index_data_list, predict_next_day_data_list_arr):
        
        detail= None
        SE= 0
        time_steps, features= predict_data.shape
        new_predict_data= predict_data.reshape(1, time_steps, features)
        prediction= model_predict(new_predict_data, model)
 
        max_val= np.max(prediction)
        test, value_index= np.where(prediction== max_val)
        if len(value_index)!= 1:
            print (predict_data_index[0], "can't predict data")
            continue
        print (value_index)
        
    return value_index

def save_preprocesss_input_data(dir, data_list):
    with open(dir, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for data in data_list:
            writer.writerow(data)

def get_preprocesss_input_data(dir):
    data_list= np.load(dir+ '.npy', allow_pickle= True)
   
    return data_list

def set_and_init_predict():
    agent_name= 'agent_phase'
    stock_id= '0050'
    task_name= 'for_proj'
    
    dir= os.getcwd()
    save_model_dir= os.path.join(dir, agent_name+'_'+ stock_id) 
    checkpoint_dir= os.path.join(save_model_dir, 'checkpoint/')

    train_data_name= agent_name+ '_train_data_'+ 'training'+ '_'+ task_name+'.csv'
    train_data_dir= os.path.join(dir , train_data_name)

    model= load_model(save_model_dir)
   

    next_day_data_name= agent_name+'_test_data_' +'testing'+ '_'+ task_name+ '.csv'
    predict_next_day_data_dir= os.path.join(dir, next_day_data_name)
    
    
    predict_next_day_data_list= get_predict_next_day_data(predict_next_day_data_dir)
    session_list, total_data_list= get_cross_train_rule_data_and_validation_data(train_data_dir)

    session_index= 0
    train_data_list= session_list[session_index][0]
    validation_data_list= session_list[session_index][1]

    train_input_data_list, train_input_data_index_list, validation_input_data_list, validation_data_input_index_list, output_train_data_index_list, output_val_data_index_list,predict_next_day_input_data_list, predict_next_day_input_data_index_list= get_input_data_for_other_input(agent_name, train_data_list, validation_data_list, total_data_list, predict_next_day_data_list) 
 

    predict_value= init_predict(model, predict_next_day_input_data_list, predict_next_day_input_data_index_list)
    
    return predict_value


def set_and_init_train():
    agent_name= 'agent_phase'
    stock_id= '0050'
    task_name= 'for_proj'
    dir= os.getcwd()
    save_model_dir= os.path.join(dir, agent_name+'_'+ stock_id) 
    checkpoint_dir= os.path.join(save_model_dir, 'checkpoint/')

    train_data_name= agent_name+ '_train_data_'+ 'training'+ '_'+ task_name+'.csv'
    train_data_dir= os.path.join(dir , train_data_name)
   

    next_day_data_name= agent_name+'_test_data_' +'testing'+ '_'+ task_name+ '.csv'
    predict_next_day_data_dir= os.path.join(dir, next_day_data_name)
    
    #testing_data = load_data(args.testing)
    memory_state= None
    carry_state= None
    
    session_list, total_data_list= get_cross_train_rule_data_and_validation_data(train_data_dir)

    session_index= 0
    train_data_list= session_list[session_index][0]
    validation_data_list= session_list[session_index][1]

    number_of_layers= 6

    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    refresh_input_data= True
    model= False

    model, (memory_state, carry_state)= init_training(agent_name, stock_id, refresh_input_data, session_list, total_data_list, number_of_layers, model, memory_state, carry_state, save_model_dir, checkpoint_dir, train_data_dir)
    
    return model


def get_outline_history_data_for_test(path):
    history_data_list= []
    reference_data_list= []
    target_data_list= []
    reference_data_dir_0= os.path.join(os.getcwd(), '0050_reference_data.csv')
    reference_data_dir_1= os.path.join(os.getcwd(), '2330_reference_data.csv')
    
    
    with open(path, 'r', newline='', encoding='utf-8-sig') as csvfile:                       
        reader = csv.reader(csvfile, delimiter=',')
        
        for row in reader:
            for col_index, col in enumerate(row):
                if col_index!= 0:
                    if col== 'null':
                        continue
                    
                    col= float(col)
                
            history_data_list.append(row)
            
    target_data_begin= history_data_list[0]
    target_data_end= history_data_list[-1]   
    target_data_begin_index= None
    target_data_end_index= None
    
    
    reference_data_dir= reference_data_dir_0
    
    with open(reference_data_dir, 'r', newline='', encoding='utf-8-sig') as csvfile:                       
        reader = csv.reader(csvfile, delimiter=',')
        data_list= [i for i in reader]
        for row in data_list[1:]:
            
            for col_index, col in enumerate(row):
                if col_index!= 0:
                    if col== 'null':
                        continue
                    col= float(col)
     
            reference_data_list.append(row)
        
        for reference_data_index, reference_data in enumerate(reference_data_list):
            if 'null' in reference_data:
                continue
                
            if [float(i) for i in target_data_begin]==[float(i) for i in reference_data[1: 5]]:               
                target_data_begin_index= reference_data_index
                
            if [float(i) for i in target_data_end]==[float(i) for i in reference_data[1: 5]]:           
                target_data_end_index= reference_data_index+ 1

    if target_data_begin_index== None and target_data_end_index== None:
    
        reference_data_dir= reference_data_dir_1 
        with open(reference_data_dir, 'r', newline='', encoding='utf-8-sig') as csvfile:                       
            reader = csv.reader(csvfile, delimiter=',')
            data_list= [i for i in reader]
            for row in data_list[1:]:
            
                for col_index, col in enumerate(row):
                    if col_index!= 0:
                        if col== 'null':
                            continue
                        col= float(col)
     
                reference_data_list.append(row)
        
            for reference_data_index, reference_data in enumerate(reference_data_list):
                if 'null' in reference_data:
                    continue
                
                if [float(i) for i in target_data_begin]==[float(i) for i in reference_data[1: 5]]:               
                    target_data_begin_index= reference_data_index
                
                if [float(i) for i in target_data_end]==[float(i) for i in reference_data[1: 5]]:           
                    target_data_end_index= reference_data_index+ 1         
    
        
    if target_data_begin_index!= None and target_data_end_index!= None:
        target_data_list =  reference_data_list[target_data_begin_index: target_data_end_index]

    
    return target_data_list

def preprocess_for_test_data(test_data_dir, target_row):
    title, ext= title, ext= os.path.splitext(os.path.basename(test_data_dir))
    stock_id= title.split('_')[0]
   # data_list= get_history_data_for_test(test_data_dir, target_row)
    data_list= get_target_row_history_data_for_test(test_data_dir, target_row)

    # Standardize
    
    # open-high-low-close
    time_column= [i[0] for i in data_list.copy()]            
    lowest_price_column= [float(i[3]) for i in data_list.copy()]
    open_price_column= [float(i[1]) for i in data_list.copy()]
            
            
    start_time= time_column[0]
    end_time= time_column[-1]
    delta_day= convert_2_day_delta(start_time, end_time)
      
    delta_price= lowest_price_column[-1]- lowest_price_column[0]
  
    slope= delta_price/ delta_day
            
            
    quartile_slope_index_list, quartile_slope_list= get_quartile_slope(data_list, time_column, lowest_price_column) # Return data index list, slope list
            
    target_quartile_slope_index_list= []
    for quartile_slope_index in range(len(quartile_slope_list)):
        if quartile_slope_list[quartile_slope_index]> slope:
            target_quartile_slope_index_list.append(quartile_slope_index)
            
    target_quartile_slope= [ quartile_slope_list[i] for i in target_quartile_slope_index_list]
    target_quartile_index_list= [quartile_slope_index_list[i] for i in target_quartile_slope_index_list]
   
    all_phase_round_range= get_all_phase_round(data_list, target_quartile_slope, target_quartile_index_list)
            
          
    price_delta_column= get_price_delta_column(data_list) # Add price delta column
    for index, content in enumerate(data_list):
                    content.append(price_delta_column[index])
                    
    data_list= get_period_data_to_previous_data_level_with_different_columns(data_list, all_phase_round_range) 
   
  
                    
    sd_level_list, sd_level_tag_column= get_sd_level(data_list, price_delta_column) 
    data_for_agent_delta_list= data_list.copy()
            
    for data_index in range(len(data_for_agent_delta_list)):
                         
        data_for_agent_delta_list[data_index].append(sd_level_tag_column[data_index]) # Add sd level column
                   
                    
    data_for_agent_phase_list= data_for_agent_delta_list.copy()       
   
    phase_stage_range_list= get_phase_stage_range(data_for_agent_phase_list)
    minimum_price_column_target_phase_peroid_range= [float(i[2]) for i in data_for_agent_phase_list]  
    open_price_column_target_phase_peroid_range= [float(i[1]) for i in data_for_agent_phase_list]  
    target_price_column_target_phase_peroid_range= open_price_column_target_phase_peroid_range
    phase_stage_column= tag_phase_stage(target_price_column_target_phase_peroid_range, phase_stage_range_list)
 
 
            
    for data_index in range(len(data_for_agent_phase_list)):
        data_for_agent_phase_list[data_index].append(phase_stage_column[data_index]) # Add phase stage column
                      
                        #data_for_agent_phase_list[data_index].append(phase_stage_range_list) # Add phase stage column list
                    
    #new_data_for_agent_phase_list+= data_for_agent_phase_list
                    
               
    agent_name= 'agent_phase_'+ 'test_data_'+ stock_id + '_for_proj'+'.csv'
    save_train_data_dir= os.path.join(os.getcwd(), agent_name)
                    
    save_train_data(save_train_data_dir, data_for_agent_phase_list) 
    
    
def get_target_row_history_data_for_test(path, target_row):
    reference_data_list= []
    target_data_list= []
    reference_data_dir_0= os.path.join(os.getcwd(), '0050_reference_data.csv')
    reference_data_dir_1= os.path.join(os.getcwd(), '2330_reference_data.csv')


    target_data_begin_index= None
    target_data_end_index= None
        
    reference_data_dir= reference_data_dir_0
    
    with open(reference_data_dir, 'r', newline='', encoding='utf-8-sig') as csvfile:                       
        reader = csv.reader(csvfile, delimiter=',')
        data_list= [i for i in reader]
        for row in data_list[1:]:
            
            for col_index, col in enumerate(row):
                if col_index!= 0:
                    if col== 'null':
                        continue
                    col= float(col)
     
            reference_data_list.append(row)
        
        for reference_data_index, reference_data in enumerate(reference_data_list):
            if 'null' in reference_data:
                continue
                
            if [float(i) for i in target_row[1:5]]==[float(i) for i in reference_data[1: 5]]:
               
                target_data_begin_index= reference_data_index- 10 # Data range= 7         
                target_data_end_index= reference_data_index+ 1


    if target_data_begin_index== None and target_data_end_index== None:
        reference_data_dir= reference_data_dir_1
        with open(reference_data_dir, 'r', newline='', encoding='utf-8-sig') as csvfile:                       
            reader = csv.reader(csvfile, delimiter=',')
            data_list= [i for i in reader]
            for row in data_list[1:]:
            
                for col_index, col in enumerate(row):
                    if col_index!= 0:
                        if col== 'null':
                            continue
                        col= float(col)
     
                reference_data_list.append(row)
        
            for reference_data_index, reference_data in enumerate(reference_data_list):
                if 'null' in reference_data:
                    continue
                
                if [float(i) for i in target_row[1:5]]==[float(i) for i in reference_data[1: 5]]:
               
                    target_data_begin_index= reference_data_index- 10 # Data range= 7         
                    target_data_end_index= reference_data_index+ 1

                
        
        
    if target_data_begin_index!= None and target_data_end_index!= None:
        target_data_list =  reference_data_list[target_data_begin_index: target_data_end_index]

    
    return target_data_list

def convert_2_day_delta(a_time, b_time):
   
    a_time_y= int(a_time.split('-')[0])
    b_time_y= int(b_time.split('-')[0])
    a_time_m= int(a_time.split('-')[1])
    b_time_m= int(b_time.split('-')[1])
    a_time_d= int(a_time.split('-')[2])
    b_time_d= int(b_time.split('-')[2])
    
    a_time_datetime = datetime.datetime(a_time_y, a_time_m, a_time_d)
    b_time_datetime= datetime.datetime(b_time_y, b_time_m, b_time_d)  
 
    delta_seconds = (b_time_datetime- a_time_datetime).total_seconds()

    delta_days= delta_seconds/ 3600
    return delta_days

def get_quartile_slope(data_list, time_column, minimum_price_column):
    quartile_slope_index_list= []
    quartile_slope_content_list= []
    target_time_index_list= []
    target_time_content_list= []
    
    
    for time_index, time in enumerate(time_column):
        
        time_y= int(time.split('-')[0])
        time_m= int(time.split('-')[1])
        time_d= int(time.split('-')[2])
        target_time_m= [1, 2, 3, 4, 5 , 6, 7, 8, 9, 10, 11, 12]     

        if time_m in target_time_m:
            begin_time_d= int(min([i[:].split('-')[2] for i in time_column if int(i[:].split('-')[1]) == time_m]))          # Get begin day in moth
            if time_d== begin_time_d:
                
                target_time_index_list.append(time_index)
                target_time_content_list.append(time)        

    target_minimum_price_list= []
    for i in target_time_index_list:
        target_minimum_price_list.append(minimum_price_column[i])
        
    for time_content_index in range(len(target_time_content_list)):
        if time_content_index+ 1== len(target_time_content_list):
            break
        a_target_time= target_time_content_list[time_content_index]
        b_target_time= target_time_content_list[time_content_index+ 1]
        
        a_target_minumum_price= target_minimum_price_list[time_content_index]
        b_target_minumum_price= target_minimum_price_list[time_content_index+ 1]
        delta_target_time= convert_2_day_delta(a_target_time, b_target_time)
        if delta_target_time== 0:
            continue
        slope= (float(b_target_minumum_price)- float(a_target_minumum_price))/ delta_target_time
    
        quartile_slope_index_list.append(a_target_time+ '_'+ b_target_time)
        quartile_slope_content_list.append(slope)
        
    return quartile_slope_index_list, quartile_slope_content_list


def get_all_phase_round(data_list, target_quartile_slope, target_quartile_index_list):
    time_column= [i[0] for i in data_list]
    target_quartile_index_list_c= target_quartile_index_list.copy()
    all_phase_round= []
    
    for target_quartile_index in range(len(target_quartile_index_list_c)):
        if target_quartile_index== 0 and len(target_quartile_index_list_c)> 1:
           
            target_start_time_index= None
            phase_round_end_time= target_quartile_index_list_c[target_quartile_index+ 1].split('_')[1]
            if not phase_round_end_time in time_column:
                continue
            tagret_end_time_index= time_column.index(phase_round_end_time)
      
            
        
        elif target_quartile_index== len(target_quartile_index_list_c)- 1:

            phase_round_start_time= target_quartile_index_list_c[target_quartile_index].split('_')[1]          
            if not phase_round_start_time in time_column:
                continue
            target_start_time_index= time_column.index(phase_round_start_time)           
            tagret_end_time_index= None
          
            
        else:
            phase_round_start_time= target_quartile_index_list_c[target_quartile_index].split('_')[1]
            phase_round_end_time= target_quartile_index_list_c[target_quartile_index+ 1].split('_')[1]
            if not phase_round_start_time in time_column or not phase_round_end_time in time_column:
                continue
            target_start_time_index= time_column.index(phase_round_start_time)
            tagret_end_time_index= time_column.index(phase_round_end_time)

        
       
        phase_round_range= data_list.copy()[target_start_time_index: tagret_end_time_index]
        
        all_phase_round.append(phase_round_range)
    return all_phase_round

def get_price_delta_column(data_list):
    delta_list= []
    for index, data in enumerate(data_list):
        if index== 0:
            delta= 0
            delta_list.append(delta)
            continue

        close_price= data[3]
        last_close_price= data_list[index- 1][3]
        delta= float(close_price)- float(last_close_price)
        delta_list.append(delta)
    
    return delta_list

def get_all_periods_data_to_previous_data_level_with_different_columns(data_list, all_phase_round_range):
    # 11: Trading Volume level (2) ; 12: Trading Volume 2 price level (3) ; 13: Top price level (5)
    # 14: Bottom price level (6); 15: Transations level (9); 16: Period 
   
    all_data_time_column_list= [i[0] for i in data_list]
    target_column_index_list= set([1, 2, 3, 4])
    target_column_mean_and_sd_val= []
    
    
    # Target data range
    m_range=   10
    week_range= 7
    
    max_range_len= max(m_range, week_range)
    for phase_round_range_index, phase_round_range in enumerate(all_phase_round_range):
        
      
        for data_index, data in enumerate(phase_round_range):
            if phase_round_range_index== 0 and data_index< max_range_len:
                continue
                
            target_data_index= all_data_time_column_list.index(data[0])
           
            for target_column_index in target_column_index_list: # Get level in target data column
                
                target_column=  [float(i[target_column_index]) for i in data_list]
                
                
                target_m_range_begin= target_data_index- m_range
                
                target_week_range_begin= target_data_index- week_range
                    
                target_column=  target_column[target_m_range_begin:target_data_index]+   target_column[target_week_range_begin:target_data_index] # Select range
     
                specific_phase_round_data_mean= statistics.mean(target_column)
          
                specific_phase_round_data_sd= statistics.stdev(target_column)
                
                
                            
                target_data= float(data[target_column_index])
              #  specific_phase_round_data_min_c= specific_phase_round_data_min
                
                data_to_previous_data= ((np.array(target_column)- np.array(target_data))/ specific_phase_round_data_sd).tolist()
             
                   

               # print ('data_to_previous_data shape', np.array(data_to_previous_data).shape)
                
                data.append(data_to_previous_data)
                
            data.append(phase_round_range_index)
        
   
            
    return all_phase_round_range, max_range_len

def sigmoid(val):
    output= 1/(1+ (math.exp(1)** -val))
    if np.isnan(output):
        output= 0
    return output



def get_period_data_to_previous_data_level_with_different_columns(data_list, all_phase_round_range):
    # 11: Trading Volume level (2) ; 12: Trading Volume 2 price level (3) ; 13: Top price level (5)
    # 14: Bottom price level (6); 15: Transations level (9); 16: Period 
    target_column_index_list= set([1, 2, 3, 4])
    target_column_mean_and_sd_val= []
    
    
    # Target data range
    m_range= 10 
    week_range= 7
    
    phase_round_range_index='n'
   
    for data_index, data in enumerate(data_list):
        for target_column_index in target_column_index_list: # Get level in target data column
                target_column=  [float(i[target_column_index]) for i in data_list if i != '']
                
                if week_range> len(target_column):
                    target_week_range_begin= 0
                else:
                    target_week_range_begin= len(target_column)- week_range
                if m_range> len(target_column):
                    target_m_range_begin= 0
                else:
                    target_m_range_begin= len(target_column)- m_range
                target_column=  target_column[target_m_range_begin:]+ target_column[target_week_range_begin:]   # Select range
                
                specific_phase_round_data_mean= statistics.mean(target_column)
              #  specific_phase_round_data_min= min(target_column)
                specific_phase_round_data_sd= statistics.stdev(target_column)
                
                target_data= float(data[target_column_index])
             
             
                    
                data_to_previous_data= ((np.array(target_column)- np.array(target_data))/ specific_phase_round_data_sd).tolist()
            
                
                data.append(data_to_previous_data)
        data.append(phase_round_range_index)
                       
            
    return data_list



def get_sd_level(data_list, target_data_level_column):
    sd_level_list= [] 
    sd_level_tag_column= []
    target_data_level_column_c= target_data_level_column.copy()
    sd_level= statistics.stdev(target_data_level_column_c)
    mean= statistics.mean(target_data_level_column)
    for i in range(-9, 10):
        r= mean+ sd_level* i
        sd_level_list.append(r)
        
    for target_data in target_data_level_column_c:
        
        level_tag= None
      
        for sd_level_index, sd_level in enumerate(sd_level_list):
          
            if sd_level_index== len(sd_level_list)-1:
                if target_data >sd_level:
                    level_tag= str(sd_level_index+ 1)
                elif target_data>= sd_level_list[sd_level_index-1] and sd_level_list[sd_level_index]<= target_data:
                    level_tag= str(sd_level_index)
                continue
             
            if sd_level_index== 0:
                if target_data <sd_level_list[sd_level_index]:
                    level_tag= str(sd_level_index)
                elif target_data >= sd_level_list[sd_level_index] and  target_data<= sd_level_list[sd_level_index+ 1]:
                    level_tag= str(sd_level_index+ 1)
                continue
                
            if sd_level_list[sd_level_index+ 1]> target_data and target_data> sd_level_list[sd_level_index] :
                level_tag= str(sd_level_index)
     
        sd_level_tag_column.append(level_tag)  
        
    return sd_level_list, sd_level_tag_column

def get_phase_stage_range(last_phase_round_range):
   
    minimum_price_list= [float(i[3]) for i in last_phase_round_range] # Lowest price column index= 3
    open_price_list= [float(i[3]) for i in last_phase_round_range]
    target_price_list= open_price_list
  
    # 8 stages
    mean= statistics.mean(target_price_list)
    minimum= min(target_price_list)- min(target_price_list)* 5/ 100 
    maximum= max(target_price_list)+ max(target_price_list)* 5/ 100
    q1, q2, q3= statistics.quantiles(target_price_list)
   
    # 1: minimum- minimum_q1_mean; 2: minimum_q1_mean- q1; 3: q1- q1_q2_mean; 4: q1_q2_mean- q2; 
    # 5: q2- q2_q3_mean; 6: q2_q3_mean- q3; 7: q3- q3_q4_mean; 8: q3_q4_mean- maximum
    minimum_q1_mean= statistics.mean([minimum, q1])
    q1_q2_mean= statistics.mean([q1, q2])
    q2_q3_mean= statistics.mean([q2, q3])
    q3_maximum_mean= statistics.mean([q3, maximum])
    
    phase_stage_list= [(minimum, minimum_q1_mean), (minimum_q1_mean, q1), (q1, q1_q2_mean), 
                       (q1_q2_mean, q2), (q2, q2_q3_mean), (q2_q3_mean, q3), 
                       (q3, q3_maximum_mean), (q3_maximum_mean, maximum)]
   
                           
    return phase_stage_list

def tag_phase_stage(target_data_column, phase_stage_range_list):
    # If price not in maximum phase stage or minimum stage, check phase round transition
    phase_stage_column= []
    for target_data in target_data_column:
        phase_tag= None
        for phase_stage_index in range(len(phase_stage_range_list)):
            lower_boundary, upper_boundary= phase_stage_range_list[phase_stage_index]
            if target_data >=lower_boundary and target_data<= upper_boundary:
                phase_tag= phase_stage_index
        if phase_tag== None:
            print ('please check phase round transition', target_data)
        phase_stage_column.append(phase_tag)
    return phase_stage_column

def save_train_data(dir, data_list):
     with open(dir, 'w', newline='') as csvfile:                       
        writer = csv.writer(csvfile)
        for data in data_list:
            writer.writerow(data)

def check_current_to_last_price(current_price, last_price):
    output= 0
    if current_price> last_price:
        output= 1

    return output
            
def get_action(data, test_data_index, predict_stage, action_history, price_history, current_holding):
    action= 0
    
    if predict_stage< 3 and current_holding== 0:
        action= 1
    elif predict_stage> 4 and current_holding== 1:
       # check_output= check_current_to_last_price(data[1], price_history[-1])
       # if check_output== 1:
        action= -1
       # else:
       #     action= 0   
        
    if action== 1:
        current_holding= 1
        price_history.append(data[1])
    if action== -1:
        current_holding= 0
        
    action_history.append(action)
    
    
    return action, action_history, price_history, current_holding

def save_action_output(data_list, action_output_dir):
   
    with open(action_output_dir, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for data in data_list:
            writer.writerow([data])    
    
     
def agent_action(test_data_dir, action_output_dir):
    
    
    data_list= get_outline_history_data_for_test(test_data_dir)
    action_history=[]
    price_history= []
    current_holding= 0
    for count, i in enumerate(data_list):
       # gc.enable()
       # gc.collect()
      #  print (i)
        if count< 10:
            action_history.append(0)
            continue
        if count== len(data_list)-1:
            break
        preprocess_for_test_data(test_data_dir, i)
        predict_value= set_and_init_predict()
        print (i)
        action, action_history, price_history, current_holding= get_action(i, count, predict_value, action_history, price_history, current_holding)
        print ('action', action, 'action_history', action_history, 'current_holding', current_holding)
    print (len(action_history))
    save_action_output(action_history, action_output_dir)
     
       
            
            




# In[5]:


if __name__ == '__main__':
    # You should not modify this part.
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
    
    # The following part is an example.
    # You can modify it at will.
    init_time= time.time()
    
    # For training 
    preprocess(args.training)
    set_and_init_train()

 
    
    # For testing
    test_data_dir= args.testing # args.testing
    action_output_dir= args.output # args.output
    agent_action(test_data_dir, action_output_dir)
    print ('Training data dir', args.training) 
    print ('Testing data dir', args.testing) 
 
    duration_time= (time.time()- init_time)/ 60
    print (f'All done in {duration_time} min.')
    
        
            
            




