
# coding: utf-8

# In[1]:

import tensorflow as tf
#tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
import numpy as np
import math
from tensorflow.contrib import slim
import os
traindir='/home/yinminghao/GW_data_MC/'
testdir='/home/yinminghao/GW_data_test_MC/'
#global_step = tf.Variable(0,trainable=False)


# In[5]:

train_epoch=6
Nreal=10000
test_size=500
batch_size=20
wave_length=8192
det_num=3
param_num=6
#parameters: inj_time, MC, sphi, stheta, (spsi?), dl0, cosi, (phi_0?)

def Conv_Class(wave_fn):
    net=slim.conv2d(wave_fn,16,[16,1],padding='VALID')
    net=slim.max_pool2d(net,[4,1])
    net=tf.nn.relu(net)
    net=slim.conv2d(net,32,[8,1],padding='VALID')
    net=slim.max_pool2d(net,[4,1])
    net=tf.nn.relu(net)
    net=slim.conv2d(net,64,[8,1],padding='VALID')
    net=slim.max_pool2d(net,[4,1])
    net=tf.nn.relu(net)
    net=slim.flatten(net)
    net=slim.fully_connected(net,64,activation_fn=None)
    net=tf.nn.relu(net)
    logits=slim.fully_connected(net,2,activation_fn=None)
    #classes=tf.nn.softmax(logits)
    return logits


def Conv_Pred(wave_fn):
    net=slim.conv2d(wave_fn,16,[16,1],padding='VALID')
    net=slim.max_pool2d(net,[4,1])
    net=tf.nn.relu(net)
    net=slim.conv2d(net,32,[8,1],padding='VALID')
    net=slim.max_pool2d(net,[4,1])
    net=tf.nn.relu(net)
    net=slim.conv2d(net,64,[8,1],padding='VALID')
    net=slim.max_pool2d(net,[4,1])
    net=tf.nn.relu(net)
    net=slim.flatten(net)
    net=slim.fully_connected(net,64,activation_fn=None)
    net=tf.nn.relu(net)
    logits=slim.fully_connected(net,param_num,activation_fn=None)
    #classes=tf.nn.softmax(logits)
    return logits

def Conv_Pred_split(wave_fn):
    
    input_1,input_2,input_3=tf.split(wave_fn,[1,1,1],3)
    
    branch1_c1=slim.conv2d(input_1,12,[16,1],padding='VALID')
    branch2_c1=slim.conv2d(input_2,12,[16,1],padding='VALID')
    branch3_c1=slim.conv2d(input_3,12,[16,1],padding='VALID')
    branch1_p1=slim.max_pool2d(branch1_c1,[4,1])
    branch2_p1=slim.max_pool2d(branch2_c1,[4,1])
    branch3_p1=slim.max_pool2d(branch3_c1,[4,1])
    branch1_a1=tf.nn.relu(branch1_p1)
    branch2_a1=tf.nn.relu(branch2_p1)
    branch3_a1=tf.nn.relu(branch3_p1)
    
    
    branch1_c2=slim.conv2d(branch1_a1,16,[12,1],padding='VALID')
    branch2_c2=slim.conv2d(branch2_a1,16,[12,1],padding='VALID')
    branch3_c2=slim.conv2d(branch3_a1,16,[12,1],padding='VALID')    
    branch1_p2=slim.max_pool2d(branch1_c2,[4,1])
    branch2_p2=slim.max_pool2d(branch2_c2,[4,1])
    branch3_p2=slim.max_pool2d(branch3_c2,[4,1])
    branch1_a2=tf.nn.relu(branch1_p2)
    branch2_a2=tf.nn.relu(branch2_p2)
    branch3_a2=tf.nn.relu(branch3_p2)
    
    net=tf.concat([branch1_a2,branch2_a2,branch3_a2],3)
    net=slim.conv2d(net,64,[8,1],padding='VALID')
    net=slim.max_pool2d(net,[4,1])
    net=tf.nn.relu(net)
    net=slim.conv2d(net,96,[6,1],padding='VALID')
    net=slim.max_pool2d(net,[4,1])
    net=tf.nn.relu(net)
    net=slim.flatten(net)
    net=slim.fully_connected(net,128,activation_fn=None)
    net=tf.nn.relu(net)
    logits=slim.fully_connected(net,param_num,activation_fn=None)
    #prediction=tf.nn.softmax(logits)
    return logits

def get_batch(batch_size,step,is_train):
    
    if is_train is True:
        datadir=traindir
    else:
        datadir=testdir
    
    wave_dir='data_bns_%s.txt'%(step*batch_size+1)
    wave_fn=np.loadtxt(datadir+wave_dir)
    wave_fn=wave_fn[:,1:det_num+1]
    #wave_fn=tf.expand_dims(wave_fn,0)
    #wave_fn=tf.expand_dims(wave_fn,2)
    wave_fn=wave_fn.reshape(1,wave_length,1,det_num)
    
    parameter_dir='injection_bns_%s.txt'%(step*batch_size+1)
    parameter_fn=np.loadtxt(datadir+parameter_dir)
    
    #parameter_fn=tf.expand_dims(parameter_fn,0)

    parameter_fn[0]=(parameter_fn[0]-0.3)*10-1
    parameter_fn[1]=(parameter_fn[1]-0.8)*4-1
    parameter_fn[3]=(parameter_fn[3]/3.1415926)-1
    parameter_fn[4]=math.sin(parameter_fn[4])
    parameter_fn[6]=(parameter_fn[6]-30)/20
    
    parameter_fn=np.delete(parameter_fn,8)
    parameter_fn=np.delete(parameter_fn,5)
    parameter_fn=np.delete(parameter_fn,2)
    
    parameter_fn=parameter_fn.reshape(1,param_num)
    
    i=1
    
    while i<batch_size:
        
        wave_dir_append='data_bns_%s.txt'%(step*batch_size+i+1)
        wave_fn_append=np.loadtxt(datadir+wave_dir_append)
        wave_fn_append=wave_fn_append[:,1:det_num+1]
        #wave_fn_append=tf.expand_dims(wave_fn_append,0)
        #wave_fn_append=tf.expand_dims(wave_fn_append,2)
        wave_fn_append=wave_fn_append.reshape(1,wave_length,1,det_num)
        
        parameter_dir_append='injection_bns_%s.txt'%(step*batch_size+i+1)
        parameter_fn_append=np.loadtxt(datadir+parameter_dir)
        
        #parameter_fn_append=tf.expand_dims(parameter_fn_append,0)
        parameter_fn_append[0]=(parameter_fn_append[0]-0.3)*10-1
        parameter_fn_append[1]=(parameter_fn_append[1]-0.8)*4-1
        parameter_fn_append[3]=(parameter_fn_append[3]/3.1415926)-1
        parameter_fn_append[4]=math.sin(parameter_fn_append[4])
        parameter_fn_append[6]=(parameter_fn_append[6]-30)/20
    
        parameter_fn_append=np.delete(parameter_fn_append,8)
        parameter_fn_append=np.delete(parameter_fn_append,5)
        parameter_fn_append=np.delete(parameter_fn_append,2)
        parameter_fn_append=parameter_fn_append.reshape(1,param_num)        

        wave_fn=np.concatenate((wave_fn,wave_fn_append),axis=0)
        parameter_fn=np.concatenate((parameter_fn,parameter_fn_append),axis=0)
        
        i=i+1
    

    
    return wave_fn,parameter_fn


def square_loss(prediction,parameter):
    loss=tf.reduce_mean(tf.square(tf.subtract(prediction,parameter)))
    return loss

def relative_square_loss(prediction,parameter):
    loss=tf.reduce_mean(tf.square(tf.subtract(prediction,parameter)/parameter))
    return loss

def cross_entropy_loss(prediction,parameter):
    cross_entropy=tf.reduce_mean(-tf.reduce_sum(parameter*tf.log(prediction),reduction_indices=[1]))
    return cross_entropy

def compute_relative_error(prediction_test,parameter_test):
    sub_all=tf.subtract(prediction_test,parameter_test)/parameter_test
    relative_error=tf.sqrt(tf.reduce_mean(tf.square(sub_all),0))
    return relative_error

def compute_mean_relative_error(prediction_test,parameter_test):
    sub_all=2*tf.subtract(prediction_test,parameter_test)/(prediction_test+parameter_test)
    mean_relative_error=tf.sqrt(tf.reduce_mean(tf.square(sub_all),0))
    return mean_relative_error

def compute_error(prediction,parameter):
    error=tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(prediction,parameter))),0)  
    return error      

# In[6]:


#merged=tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
#writer=tf.summary.FileWriter("/logs",sess.graph)

wave_holder=tf.placeholder(tf.float32,shape=[None,wave_length,1,det_num])
parameter_holder=tf.placeholder(tf.float32,shape=[None,param_num])

prediction=Conv_Pred(wave_holder)

loss=square_loss(prediction,parameter_holder)

relative_error=compute_relative_error(prediction,parameter_holder)
error=compute_error(prediction,parameter_holder)

#train_op=tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(loss)
train_op=tf.train.AdamOptimizer(learning_rate=1e-2).minimize(loss)
#train_op=tf.train.AdadeltaOptimizer(learning_rate=1e-2).minimize(loss)
#train_op=tf.train.MomentumOptimizer(learning_rate=1e-2,momentum=0.9).minimize(loss)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

init=tf.initialize_all_variables()

#init_fn=
sess.run(init)
#saver=tf.train.Saver()
epoch=0
while epoch<train_epoch:
    
    step=0
    file1=open('/home/yinminghao/logs/normalize_relativeerror_epoch_%s.txt'%epoch,'w')
    file2=open('/home/yinminghao/logs/normalize_error_epoch_%s.txt'%epoch,'w')
    while step<(Nreal/batch_size):
        wave_batch,parameter_batch=get_batch(batch_size,step,is_train=True)
        _,loss_value=sess.run([train_op,loss],feed_dict={wave_holder:wave_batch,parameter_holder:parameter_batch})
        #tf.summary.scalar("loss",loss_value)
        #result=sess.run(merged)#,feed_dict={wave_holder:wave_batch,parameter_holder:parameter_batch})
        #writer.add_run_metadata(run_metadata,'step%03d'%(epoch*(Nreal/batch_size)+step))
        #writer.add_summary(result,epoch*(Nreal/batch_size)+step)
        
        
        if step%10==0:
            wave_test_batch,parameter_test_batch=get_batch(test_size,0,is_train=False)
            relative_error_value,error_value=sess.run(
                [relative_error,error],feed_dict={wave_holder:wave_test_batch,parameter_holder:parameter_test_batch})
            
            a=relative_error_value
            b=error_value
            
            format_str=('step=%d,rela_error:%.4f %.4f %.4f %.4f %.4f %.4f ,loss=%f')
            context=format_str%(step,a[0],a[1],a[2],a[3],a[4],a[5],loss_value)+'\n'
            file1.write(context)

            format_str=('step=%d,error:%.4f %.4f %.4f %.4f %.4f %.4f ,loss=%f')
            context=format_str%(step,b[0],b[1],b[2],b[3],b[4],b[5],loss_value)+'\n'
            file2.write(context)
            
            format_str=('epoch=%d,step=%d,rela_error=%s,error=%s,loss=%f')
            context=format_str%(epoch,step,str(a),str(b),loss_value)
            print(context)
        
        step=step+1
        
        
    epoch=epoch+1
        
#save_path=saver.save(sess,"/logs/GWmodel_train.ckpt")    
    
    


    


# In[ ]:




# In[38]:


# In[ ]:




