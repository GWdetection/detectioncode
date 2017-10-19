
# coding: utf-8

# In[82]:

import torch
from torch.autograd import Variable
import numpy as np
import math
from torch.nn.functional import relu
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
datadir='C:/Program Files/MATLAB/R2016b/bin/'
testdir='C:/Program Files/MATLAB/R2016b/bin/'


# In[83]:

def forward(inj_time,Mc,sphi,stheta,dl0,cosi):
    det_num=3
    wave_length=8192
    noise_scale=8.0e-22
    pi=3.1415926
    G=6.673e-11
    c=299792458.0
    Mpc=3.08567758e22
    Msun = 1.989e30
    fs=8192
    T=1
    nsamples=T*fs
    t=np.arange(nsamples)/fs

    Det1_V=np.array([-2.161414928e+06,-3.834695183e+06,4.600350224e+06])
    Det2_V=np.array([-7.427604192e+04,-5.496283721e+06,3.224257016e+06])
    Det3_V=np.array([4546374.0,842990.0,4378577.0])
    Det1_d=np.array([[-0.392614701790361,-0.077612252813702,-0.247388405118613],
                     [-0.077612252813702,0.319524089053145,0.227998293910978],
                     [-0.247388405118613,0.227998293910978,0.073090613199948]])
    Det2_d=np.array([[0.411281743683125,0.140209630402064,0.247293475274344],
                     [0.140209630402064,-0.109005942619247,-0.181616030843724],
                     [0.247293475274344,-0.181616030843724,-0.302275800865383]])
    Det3_d=np.array([[0.243874678248284,-0.099086615422263,-0.232575796255783],
                     [-0.099086615422263,-0.447827871578090,0.187828534783639],
                     [-0.232575796255783,0.187828534783639,0.203953193329806]])
    
    
    
    dl=dl0*Mpc

    m1=torch.sin(sphi)*torch.cos(spsi)-torch.cos(sphi)*torch.cos(stheta)*torch.sin(spsi)
    m2=-torch.cos(sphi)*torch.cos(spsi)-torch.sin(sphi)*torch.cos(stheta)*torch.sin(spsi)
    m3=torch.sin(stheta)*torch.sin(spsi)
    n1=-torch.sin(sphi)*torch.sin(spsi)-torch.cos(sphi)*torch.cos(stheta)*torch.cos(spsi) 
    n2=torch.cos(sphi)*torch.sin(spsi)-torch.sin(sphi)*torch.cos(stheta)*torch.cos(spsi) 
    n3 =torch.sin(stheta)*torch.cos(spsi) 
    mm=torch.cat((m1*m1,m1*m2,m1*m3,m2*m1,m2*m2,m2*m3,m3*m1,m3*m2,m3*m3),0)
    mn=torch.cat((m1*n1,m1*n2,m1*n3,m2*n1,m2*n2,m2*n3,m3*n1,m3*n2,m3*n3),0)
    nm=torch.cat((n1*m1,n1*m2,n1*m3,n2*m1,n2*m2,n2*m3,n3*m1,n3*m2,n3*m3),0)
    nn=torch.cat((n1*n1,n1*n2,n1*n3,n2*n1,n2*n2,n2*n3,n3*n1,n3*n2,n3*n3),0)
    e_plus=mm-nn
    e_cross=mn+nm
    d1=torch.from_numpy(Det1_d.reshape(9))
    d2=torch.from_numpy(Det2_d.reshape(9))
    d3=torch.from_numpy(Det3_d.reshape(9))
    Fp1=torch.sum(e_plus*Variable(d1))
    Fx1=torch.sum(e_cross*Variable(d1))
    Fp2=torch.sum(e_plus*Variable(d2))
    Fx2=torch.sum(e_cross*Variable(d2))
    Fp3=torch.sum(e_plus*Variable(d3))
    Fx3=torch.sum(e_cross*Variable(d3))
    
    
    
    omega = torch.cat((torch.sin(stheta)*torch.cos(sphi), torch.sin(stheta)*torch.sin(sphi), torch.cos(stheta)),0)

    delay_1=-torch.sum(Variable(torch.from_numpy(Det1_V))*omega)/c
    delay_2=-torch.sum(Variable(torch.from_numpy(Det2_V))*omega)/c
    delay_3=-torch.sum(Variable(torch.from_numpy(Det3_V))*omega)/c
    tc1=inj_time+delay_1
    tc2=inj_time+delay_2
    tc3=inj_time+delay_3
    idinjt1=torch.ceil(tc1*fs)
    idinjt2=torch.ceil(tc2*fs)
    idinjt3=torch.ceil(tc3*fs)
    
    npbase=np.arange(fs)/fs
    base=Variable(torch.from_numpy(npbase))
    tau1=tc1.expand(base.size())-base
    tau2=tc2.expand(base.size())-base
    tau3=tc3.expand(base.size())-base
    #tau1_relu=0.5*(torch.sign(tau1)+1)
    #tau2_relu=0.5*(torch.sign(tau2)+1)
    #tau3_relu=0.5*(torch.sign(tau3)+1)
    
    
    tau1_phi=(torch.pow(relu(tau1),5/8))
    phi_t1=-2*torch.pow((5*G*Mc/(c*c*c)),-5/8).expand(tau1_phi.size())*tau1_phi
    tau1_Ah=torch.pow(relu(5/(c*tau1)),1/4)
    Ah1=(1/dl.expand(tau1_Ah.size()))*torch.pow(G*Mc/(c*c),5/4).expand(tau1_Ah.size())*tau1_Ah
    hp1=0.5*(1+torch.pow(cosi,2.0)).expand(Ah1.size())*(Ah1*torch.cos(phi_t1).expand(Ah1.size()))
    hx1=Ah1*torch.cos(phi_t1).expand(Ah1.size())*cosi.expand(Ah1.size())

    tau2_phi=torch.pow(relu(tau2),5/8)
    phi_t2=-2*torch.pow((5*G*Mc/(c*c*c)),-5/8).expand(tau2_phi.size())*tau2_phi
    tau2_Ah=torch.pow(relu(5/(c*tau2)),1/4)
    Ah2=(1/dl.expand(tau2_Ah.size()))*torch.pow(G*Mc/(c*c),5/4).expand(tau2_Ah.size())*tau2_Ah
    hp2=0.5*(1+torch.pow(cosi,2.0)).expand(Ah2.size())*(Ah2*torch.cos(phi_t2).expand(Ah2.size()))
    hx2=Ah2*torch.cos(phi_t2).expand(Ah2.size())*cosi.expand(Ah2.size())

    tau3_phi=torch.pow(relu(tau3),5/8)
    phi_t3=-2*torch.pow((5*G*Mc/(c*c*c)),-5/8).expand(tau3_phi.size())*tau3_phi
    tau3_Ah=torch.pow(relu(5/(c*tau3)),1/4)
    Ah3=(1/dl.expand(tau3_Ah.size()))*torch.pow(G*Mc/(c*c),5/4).expand(tau3_Ah.size())*tau3_Ah
    hp3=0.5*(1+torch.pow(cosi,2.0)).expand(Ah3.size())*(Ah3*torch.cos(phi_t3).expand(Ah3.size()))
    hx3=Ah3*torch.cos(phi_t3).expand(Ah3.size())*cosi.expand(Ah3.size())
    
    Wave1=Fp1.expand(hp1.size())*hp1+Fx1.expand(hx1.size())*hx1
    Wave2=Fp2.expand(hp2.size())*hp2+Fx2.expand(hx2.size())*hx2
    Wave3=Fp3.expand(hp3.size())*hp3+Fx3.expand(hx3.size())*hx3
    Wave=torch.cat((Wave1,Wave2,Wave3),0).view(-1,fs)
    
    return Wave




# In[84]:

def get_data():
    wave_dir='data_bns_1.txt'
    wave_fn=np.loadtxt(datadir+wave_dir)
    wave_fn=wave_fn[:,1:4]
    parameter_dir='injection_bns_1.txt'
    parameter_fn=np.loadtxt(datadir+parameter_dir)
    
    return wave_fn,parameter_fn


# In[85]:

pi=3.1415926
inj_time=Variable(torch.DoubleTensor([0.4]),requires_grad=True)
Mc=Variable(torch.DoubleTensor([2.375201545839378e+30]))
sphi=Variable(torch.DoubleTensor([pi]),requires_grad=True)
stheta=Variable(torch.DoubleTensor([0.0]),requires_grad=True)
spsi=Variable(torch.DoubleTensor([45*pi/180]))
dl0=Variable(torch.DoubleTensor([30.0]),requires_grad=True)
cosi=Variable(torch.DoubleTensor([0.5]),requires_grad=True)
phi_0=Variable(torch.DoubleTensor([0.0]))



# In[86]:

Wave_data,params_data=get_data()
Wave_data=Variable(torch.DoubleTensor(Wave_data))
params_data=Variable(torch.DoubleTensor(params_data))


# In[87]:

optimizer=torch.optim.SGD([inj_time,sphi,stheta,dl0,cosi],lr=0.1)
noise_scale=8.0e-22
for i in range(100):
    optimizer.zero_grad()
    Wave=forward(inj_time,Mc,sphi,stheta,dl0,cosi)
    loss=1e19*torch.sqrt(torch.sum(torch.pow(Wave-Wave_data,2)))
    loss1=1e38*torch.sum(torch.abs(torch.pow((Wave-Wave_data),2)-noise_scale*noise_scale))
    loss1.backward()
    optimizer.step()
    if i%10==0:
        print(loss.data)
        print(inj_time.data)


# In[ ]:

print(cosi)


# In[89]:

print(params_data)


# In[ ]:



