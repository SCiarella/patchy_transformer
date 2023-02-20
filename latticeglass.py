import torch
import sys
import random
import numpy as np
import multiprocessing as mp
import jax.numpy as jnp




def read_sample(filename):
    with open(filename) as f:
        lines=f.readlines()
        Nsamples = int(len(lines)/2)
        Nsites= len(lines[1])-1
        sample = torch.zeros(size=(Nsamples,Nsites))
        idx=0
        for line in lines:
            if line[0]=='>':
                continue
            else:
                sample[idx]=torch.Tensor([int(x) for x in line[:-1]])
                idx+=1
    return sample


def energy(inputs, nn, ncolors):
    inputs = torch.nn.functional.one_hot(inputs.to(torch.int64),num_classes=ncolors).cpu().detach().numpy()
    sum_nn = inputs[:,nn[0],:]+inputs[:,nn[1],:]+inputs[:,nn[2],:]+inputs[:,nn[3],:]+inputs[:,nn[4],:]+inputs[:,nn[5],:]
    sum_nn = sum_nn[:,:,1]+sum_nn[:,:,2]
    occ = inputs[:,:,1]+inputs[:,:,2]
    ell = jnp.einsum('nia,nia->ni',inputs,jnp.full(inputs.shape,jnp.array([0,3,5])))
    energy = occ * jnp.power(sum_nn - ell, 2)
    energy = energy.sum(axis=1)
    return energy

def patch_energy(inputs, mask, nn, ncolors):
    inputs = torch.nn.functional.one_hot(inputs.to(torch.int64),num_classes=ncolors).cpu().detach().numpy()
    sum_nn = inputs[:,nn[0],:]+inputs[:,nn[1],:]+inputs[:,nn[2],:]+inputs[:,nn[3],:]+inputs[:,nn[4],:]+inputs[:,nn[5],:]
    sum_nn = sum_nn[:,:,1]+sum_nn[:,:,2]
    occ = (inputs[:,:,1]+inputs[:,:,2])*mask
    ell = jnp.einsum('nia,nia->ni',inputs,jnp.full(inputs.shape,jnp.array([0,3,5])))
    energy = occ * jnp.power(sum_nn - ell, 2)
    energy = energy.sum(axis=1)
    return energy

def make_nn(s):
    sites = jnp.array(range(s**3)) 
    ix = sites % s
    iy = (sites//s % s)
    iz = sites//s**2
    nn1 = (ix+1)%s + s*iy + s*s*iz
    nn2 = (ix-1)%s + s*iy + s*s*iz
    nn3 = ix + s*((iy+1)%s) + s*s*iz
    nn4 = ix + s*((iy-1)%s) + s*s*iz
    nn5 = ix + s*iy + s*s*((iz+1)%s)
    nn6 = ix + s*iy + s*s*((iz-1)%s)
    return jnp.stack((nn1,nn2,nn3,nn4,nn5,nn6))


def read_sample_for_cnn(filename, q):
    with open(filename) as f:
        lines=f.readlines()
        Nsamples = int(len(lines)/2)
        Nsites= len(lines[1])-1
        L = round(np.power(Nsites,1/3))
        Lsq = L*L
        #print('N={} --> L={}'.format(Nsites,L))
        sample = torch.zeros(size=(Nsamples,q,L,L,L))
        idx=0
        for line in lines:
            if line[0]=='>':
                continue
            else:
                for i,q in enumerate(line[:-1]):
                    i = int(i)
                    z = int(i/Lsq)
                    y = int((i-z*Lsq)/L) 
                    x = i-z*Lsq-y*L
                #    print('idx={} ----> ({},{},{})\t value={}'.format(i,x,y,z,q))
                    sample[idx,int(q),x,y,z]=1
                idx+=1
    return sample


def convert_cnn_samples_to_1d(cnnsample):
    Nsamples = cnnsample.shape[0]
    q = cnnsample.shape[1]
    L = cnnsample.shape[2]
    Nsites=L*L*L
    Lsq=L*L
    onedsample = torch.zeros(size=(Nsamples,Nsites))
    for x in range(L):
        for y in range(L):
            for z in range(L):
                s = x + L*y + Lsq*z
                for qi in range(1,q):
                    onedsample[:, s] += qi*cnnsample[:, qi, x, y, z]
    return onedsample

# this version is without one-hot-encoding
def convert_3d_samples_to_1d(cnnsample, is_torch=True):
    Nsamples = cnnsample.shape[0]
    L = cnnsample.shape[2]
    Nsites=L*L*L
    Lsq=L*L
    if is_torch:
        onedsample = torch.zeros(size=(Nsamples,Nsites))
    else:
        onedsample = np.zeros(shape=(Nsamples,Nsites))
    for x in range(L):
        for y in range(L):
            for z in range(L):
                s = x + L*y + Lsq*z
                onedsample[:, s] = cnnsample[:, x, y, z]
    return onedsample

# Function to extract a patch from a full configuration
def patch_from_config_not_ordered(conf,center_id, L, is_torch=True, apply_transformation=False, transf_id = []):
    Lsq=L*L
    Lminone=L-1
    know_transformations=13
    # given the center of the patch I have to include a cube of size 5x5x5 centered around it
    # * notice that I extract the patch following the order that the model wants 

    # First I make the sample in 3d so it is easier to get the patch
    if is_torch:
        config_3d = torch.ones(size=(L,L,L))
    else:
        config_3d = np.ones(shape=(L,L,L))
    for x in range(L):
        for y in range(L):
            for z in range(L):
                s = x + L*y + Lsq*z
                config_3d[x,y,z]=conf[s]

    # since L=10 I can use the decimal number of the ID to get the position
    zc, yc, xc = str(center_id).zfill(3)
    xc = int(xc)
    yc = int(yc)
    zc = int(zc)

    if is_torch:
        patch_3d = torch.zeros(size=(1,5,5,5))
        natural_idx_correspondence = torch.zeros(size=(1,5,5,5),dtype=torch.int)
    else:
        patch_3d = np.zeros(shape=(1,5,5,5))
    for zp in range(5):
        for yp in range(5):
            for xp in range(5):
                x = xc + (xp - 2)
                y = yc + (yp - 2)
                z = zc + (zp - 2)
                # and satisfy PBC
                if x<0:
                    x+=L
                if x>Lminone:
                    x-=L
                if y<0:
                    y+=L
                if y>Lminone:
                    y-=L
                if z<0:
                    z+=L
                if z>Lminone:
                    z-=L
                
                patch_3d[:,xp,yp,zp] = config_3d[x,y,z]
                if is_torch:
                    natural_idx_correspondence[:,xp,yp,zp] = z*Lsq + y*L + x

    # *** Apply a patch transformation if required 
    if apply_transformation:
        # I have a list of transformations that mantains the energy of a 3x3x3 patch
        # (0) [n=1] original state
        # (1) [n=3x3] each direction can rotate in 3 different states 
        # (3) [n=3] I can cut halfway along each direction and flip 
        # (4) [n=3x2] I can cut diagonally along each direction and flip 
        # (5) ... there are probably some others that I did not implement
        transf_conf = patch_3d.copy()

        if transf_id>0 and transf_id<10:
            # This means that we have to do a rotation, but which axis?
            if transf_id<4:
                # x-rotation, but how many times?
                if transf_id==1:
                    cos_t=0
                    sin_t=1
                if transf_id==2:
                    cos_t=-1
                    sin_t=0
                if transf_id==3:
                    cos_t=0
                    sin_t=-1
                R = np.asarray([[1,0,0],[0,cos_t,-sin_t],[0,sin_t,cos_t]])
            elif transf_id<7:
                # y-rotation, but how many times?
                if transf_id==4:
                    cos_t=0
                    sin_t=1
                if transf_id==5:
                    cos_t=-1
                    sin_t=0
                if transf_id==6:
                    cos_t=0
                    sin_t=-1
                R = np.asarray([[cos_t,0,sin_t],[0,1,0],[-sin_t,0,cos_t]])
            elif transf_id<10:
                # z-rotation, but how many times?
                if transf_id==7:
                    cos_t=0
                    sin_t=1
                if transf_id==8:
                    cos_t=-1
                    sin_t=0
                if transf_id==9:
                    cos_t=0
                    sin_t=-1
                R = np.asarray([[cos_t,-sin_t,0],[sin_t,cos_t,0],[0,0,1]])
            else:
                print('* Error: tranf={} is not a rotation'.format(transf_id))
                sys.exit()
            # actually apply the rotation
            for z in range(5):
                for y in range(5):
                    for x in range(5):
                        xi,yi,zi = R*[x,y,z]
                        transf_conf[:,xi,yi,zi]=patch_3d[:,x,y,z]
        elif transf_id==10:
            # flip along plane perpendicolar to x
            for z in range(5):
                for y in range(5):
                    transf_conf[:,0,y,z]=patch_3d[:,4,y,z]
                    transf_conf[:,1,y,z]=patch_3d[:,3,y,z]
                    transf_conf[:,4,y,z]=patch_3d[:,0,y,z]
                    transf_conf[:,3,y,z]=patch_3d[:,1,y,z]
        elif transf_id==11:
            # flip along plane perpendicolar to y
            for z in range(5):
                for x in range(5):
                    transf_conf[:,x,0,z]=patch_3d[:,x,4,z]
                    transf_conf[:,x,1,z]=patch_3d[:,x,3,z]
                    transf_conf[:,x,4,z]=patch_3d[:,x,0,z]
                    transf_conf[:,x,3,z]=patch_3d[:,x,1,z]
        elif transf_id==12:
            # flip along plane perpendicolar to x
            for x in range(5):
                for y in range(5):
                    transf_conf[:,x,y,0]=patch_3d[:,x,y,4]
                    transf_conf[:,x,y,1]=patch_3d[:,x,y,3]
                    transf_conf[:,x,y,4]=patch_3d[:,x,y,0]
                    transf_conf[:,x,y,3]=patch_3d[:,x,y,1]
        elif transf_id>know_transformations:
            print('\n*** Error: I do not know {} transformations'.format(transf_id))
            sys.exit()
        patch_3d = transf_conf

    if is_torch:
        return convert_3d_samples_to_1d(patch_3d), convert_3d_samples_to_1d(natural_idx_correspondence).squeeze().to(dtype=torch.int)
    else:
        return convert_3d_samples_to_1d(patch_3d,False)

