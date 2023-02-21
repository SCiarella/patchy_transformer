import pandas as pd
import time
import sys
import numpy as np
import os
import torch
import torchvision
from numpy import sqrt
import math
from torch import nn
import latticeglass
from args import args
from resmade import MADE
from utils import (
    clear_checkpoint,
    clear_log,
    get_last_checkpoint_step,
    ignore_param,
    init_out_dir,
    my_log,
    print_args,
    EarlyStopping,
    mysorting,
)
import itertools
import plotly.express as px
import math
from transformer import Transformer
from torch.nn import functional as F


def main():
    start_time = time.time()

    init_out_dir()
    if args.clear_checkpoint:
        clear_checkpoint()
    last_step = get_last_checkpoint_step()
    if last_step >= 0:
        my_log('\nCheckpoint found: {}\n'.format(last_step))
    else:
        clear_log()
    print_args()
    with open('{}.myargs'.format(args.out_filename), 'w') as f:
        print_args(f.write)

    args.L=int(round(np.power(args.n,1/3)))
    args.patch_size = int(args.L*0.5)
    boundary_size = 5**3 - 3**3


    ntokens = 5  # size of vocabulary
    max_output_length = 29 #extra end of word token
    net = Transformer(ntokens, max_output_length, 2**10).to(args.device)
    net.to(args.device)
    my_log('{}\n'.format(net))

    params = list(net.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = int(sum([np.prod(p.shape) for p in params]))
    my_log('Total number of trainable parameters: {}'.format(nparams))
    named_params = list(net.named_parameters())

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr)
    elif args.optimizer == 'sgdm':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=args.lr, alpha=0.99)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))
    elif args.optimizer == 'adam0.5':
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.5, 0.999))
    else:
        raise ValueError('Unknown optimizer: {}'.format(args.optimizer))

    if args.lr_schedule:
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #    #optimizer, factor=0.5, patience=500, threshold=1e-4, min_lr=1e-5)
        #    optimizer, factor=0.92, patience=500, threshold=1e-4, min_lr=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10000, 0)

    if args.early_stopping:
        my_log('INFO: Initializing early stopping')
        early_stopping = EarlyStopping(patience=5000)

    if last_step >= 0:
        state = torch.load('{}_save/{}.state'.format(args.out_filename, last_step),map_location=torch.device(args.device))
        ignore_param(state['net'], net)
        net.load_state_dict(state['net'])
        if state.get('optimizer'):
            optimizer.load_state_dict(state['optimizer'])
        if args.lr_schedule and state.get('scheduler'):
            scheduler.load_state_dict(state['scheduler'])


    init_time = time.time() - start_time
    my_log('init_time = {:.3f}'.format(init_time))


    my_log('\n*** Patch size {} from a box of size {}'.format(args.patch_size,args.L))
    nn_list_full  = latticeglass.make_nn(args.L)
    nn_list_patch = latticeglass.make_nn(args.patch_size)


    #***********************************************************************************************************
    #******************************************   IMPORT DATA FILES ********************************************
    with torch.no_grad():
        my_log('%%%% Loading the full sample')

        rawfilename = 'LG_eq_configs_T_{}.txt'.format(args.T)
        full_sample = latticeglass.read_sample(rawfilename)
        
        # Now I measure the Energy of the full sample 
        samplefilename = 'Eequi_T{}_L{}_ps{}.npy'.format(args.T,args.L,args.patch_size)
        if os.path.isfile(samplefilename):
            E_equi = np.load(samplefilename)
        else:
            my_log('Calculating the energy')
            full_sample_energy = torch.Tensor(np.asarray(latticeglass.energy(full_sample, nn_list_full, args.q)))
            E_equi =full_sample_energy.mean()/(args.L**3)
            np.save(samplefilename,E_equi)
        my_log('*** Equilibrium E/N={}'.format(E_equi))

        # pass to numpy which is faster to operate in parallel
        full_sample = full_sample.to(dtype=int).detach().cpu().numpy()

        # and then the energy of the patches
        # ***[!]*** The patches are NOT defined in PBC, so to get their bulk energy I have to exclude the boundaries
        # to do so I define a mask which is 0 on the boundaries
        mask_patch= torch.ones(size=(1,args.patch_size,args.patch_size,args.patch_size))
        mask_patch[:,0,:,:]=0
        mask_patch[:,args.patch_size-1,:,:]=0
        mask_patch[:,:,0,:]=0
        mask_patch[:,:,args.patch_size-1,:]=0
        mask_patch[:,:,:,0]=0
        mask_patch[:,:,:,args.patch_size-1]=0
        onedmask = torch.zeros(size=(1,args.patch_size**3))
        for x in range(args.patch_size):
            for y in range(args.patch_size):
                for z in range(args.patch_size):
                    s = x + args.patch_size*y + args.patch_size*args.patch_size*z
                    onedmask[:, s] = mask_patch[:, x, y, z]
        # save the mask
        samplefilename = 'onedmask_T{}_L{}_ps{}.npy'.format(args.T,args.L,args.patch_size)
        np.save(samplefilename,onedmask)

        # I want to impose a different ordering for the variables in the patches that has the following requisites:
        #     1) first I put the fixed boundaries
        #     2) the first (mobile) spin is the center of the patch
        #     3) then we have the center of the faces (1boundary connection)
        #     4) then the edges (touching 2 boundaries)
        #     5) finally the corners
        # I can achieve this similarly to what I did in the coloring for the ordering based on the graph
        # So I create an array of all the IDs in the order that I want 
        order_of_sites=[]
        # (1)
        for x in [0,args.patch_size-1]:
            for y in range(args.patch_size):
                for z in range(args.patch_size):
                    s = x + args.patch_size*y + args.patch_size*args.patch_size*z
                    order_of_sites.append(s)
        for x in range(args.patch_size):
            for y in [0,args.patch_size-1]:
                for z in range(args.patch_size):
                    s = x + args.patch_size*y + args.patch_size*args.patch_size*z
                    order_of_sites.append(s)
        for x in range(args.patch_size):
            for y in range(args.patch_size):
                for z in [0,args.patch_size-1]:
                    s = x + args.patch_size*y + args.patch_size*args.patch_size*z
                    order_of_sites.append(s)
        # (2)
        x = 2
        y = 2
        z = 2
        s = x + args.patch_size*y + args.patch_size*args.patch_size*z
        order_of_sites.append(s)
        # (3)
        y = 2
        z = 2
        for x in [1,3]:
            s = x + args.patch_size*y + args.patch_size*args.patch_size*z
            order_of_sites.append(s)
        x = 2
        z = 2
        for y in [1,3]:
            s = x + args.patch_size*y + args.patch_size*args.patch_size*z
            order_of_sites.append(s)
        x = 2
        y = 2
        for z in [1,3]:
            s = x + args.patch_size*y + args.patch_size*args.patch_size*z
            order_of_sites.append(s)
        # (4)
        x = 2
        for y in [1,3]:
            for z in [1,3]:
                s = x + args.patch_size*y + args.patch_size*args.patch_size*z
                order_of_sites.append(s)
        y = 2
        for x in [1,3]:
            for z in [1,3]:
                s = x + args.patch_size*y + args.patch_size*args.patch_size*z
                order_of_sites.append(s)
        z = 2
        for x in [1,3]:
            for y in [1,3]:
                s = x + args.patch_size*y + args.patch_size*args.patch_size*z
                order_of_sites.append(s)
        # (5)
        for x in [1,3]:
            for y in [1,3]:
                for z in [1,3]:
                    s = x + args.patch_size*y + args.patch_size*args.patch_size*z
                    order_of_sites.append(s)
        # remove repetitions
        order_of_sites = list(dict.fromkeys(order_of_sites)) 
        my_log('## I ordered the sites in the following way:\n{}'.format(order_of_sites))
        
        # I also need the inverse order which will be used to revert back the generated samples and compute their energy
        reverse_order = np.arange(args.patch_size**3) 
        reverse_order = [x for _,x in sorted(zip(order_of_sites,reverse_order), key=lambda pair: pair[0])]
    
        my_log('storing info')
        samplefilename = 'ordering_T{}_L{}_ps{}.npy'.format(args.T,args.L,args.patch_size)
        np.save(samplefilename,order_of_sites)


    train_sample = full_sample[args.batch_size:]
    val_sample = full_sample[:args.batch_size]#.to(args.device)

    my_log('Preparing validation sample...')
    # prepare the validation sample once
    val_patches = torch.zeros(size=(args.batch_size, args.patch_size**3))
    # ** (1) select patch center
    centers = np.random.choice(np.arange(len(val_sample[0])),size=(args.batch_size,1)).astype(int)
    # ** (2) select how to transform the patch
    know_transformations = 13
    transformation = np.random.choice(np.arange(know_transformations),size=(args.batch_size,1)).astype(int)
    csamp = np.concatenate((centers,transformation,val_sample),axis=-1)
    # ** (3) extract the patch and transform it
    f = lambda x: latticeglass.patch_from_config_not_ordered(x[2:],x[0],args.L,False, True, x[1])
    val_patches = torch.Tensor(np.apply_along_axis(f, 1, csamp).squeeze(axis=1))
    patch_energy = torch.Tensor(np.asarray(latticeglass.patch_energy(val_patches, onedmask.cpu().detach().numpy(), nn_list_patch, args.q)))
    my_log('The validation patches have E/N={}'.format(patch_energy.mean()/3**3))
    # ** (4) sort the validation sample
    for sample_i in range(val_patches.shape[0]):
        val_patches[sample_i] =  mysorting(order_of_sites,val_patches[sample_i])
    # ** (5) measure the density (that I am planning to conserve)
    zeros= (val_patches == 0).sum(dim=-1).unsqueeze(dim=-1)
    ones = (val_patches == 1).sum(dim=-1).unsqueeze(dim=-1)
    twos = (val_patches == 2).sum(dim=-1).unsqueeze(dim=-1)
    Npatch = 5**3
    Nall = args.batch_size*Npatch
    #val_sample_wrho = torch.concat([zeros/Npatch,ones/Npatch,twos/Npatch,torch.Tensor(val_patches)],dim=-1)
    val_sample_wrho = torch.concat([zeros/Npatch,ones/Npatch,twos/Npatch,val_patches],dim=-1)
    # ** (6) pass the sample to deice for faster processing
    val_sample_wrho = val_sample_wrho.to(device=args.device)

    # create start and end tokens
    start_token = torch.zeros(size=zeros.shape).to(device=args.device) 
    end_token = start_token + 1

    criterion = nn.CrossEntropyLoss()

    my_log('Training...')
    newsample_resorted = torch.zeros(size=(args.batch_size,5**3))
    oldsample_resorted = torch.zeros(size=(args.batch_size,5**3))
    sample_time = 0
    train_time = 0
    start_time = time.time()



    # **********************************************
    # **************** TRAINING LOOP 
    # **********************************************
    for step in range(last_step + 1, args.max_step + 1):

        # Use the annealing rate same as Wu et al. to decrease T
        beta = 1/args.T * (1 - args.beta_anneal**step)

        #  **** Maximum likelihood
        if args.ARtype == 'maxlike':
            sample_start_time = time.time()

            # **** Batch preparation
            # * (1) extract random configs
            indices = np.random.choice(np.arange(train_sample.shape[0]),size=(args.batch_size))
            sample = train_sample[indices][:]
            # * (2) select patch center
            centers = np.random.choice(np.arange(len(sample[0])),size=(args.batch_size,1)).astype(int)
            # * (3) select how to transform the patch
            transformation = np.random.choice(np.arange(know_transformations),size=(args.batch_size,1)).astype(int)
            csamp = np.concatenate((centers,transformation,sample),axis=-1)
            # * (4) extract the patch and transform it
            f = lambda x: latticeglass.patch_from_config_not_ordered(x[2:],x[0],args.L,False, True, x[1])
            patches = torch.Tensor(np.apply_along_axis(f, 1, csamp).squeeze(axis=1))
            # * (5) sort the training sample
            for sample_i in range(args.batch_size):
                patches[sample_i] =  mysorting(order_of_sites,patches[sample_i])
            # * (6) measure the density (that I am planning to conserve)
            zeros= (patches == 0).sum(dim=-1).unsqueeze(dim=-1).to(device=args.device)
            ones = (patches == 1).sum(dim=-1).unsqueeze(dim=-1).to(device=args.device)
            twos = (patches == 2).sum(dim=-1).unsqueeze(dim=-1).to(device=args.device)
                
            patches = patches.to(device=args.device)

            sample_wrho = torch.concat([start_token,zeros/Npatch,ones/Npatch,twos/Npatch,patches],dim=-1).to(device=args.device)

            # Teacher forcing: use end and start tokens
            data = patches[:,:98]
            #data = torch.concat([data+2,end_token],dim=-1).to(device=args.device,dtype=torch.long)
            data = data.to(device=args.device,dtype=torch.long)+2
            target = patches[:,98:]
            target = torch.concat([start_token,target+2,end_token],dim=-1).to(device=args.device,dtype=torch.long)




            if args.print_step and step % args.print_step == 0 :
                # generate a new sample to evaluate the energy that the network can generate and evalaute the acceptance
                with torch.no_grad():
                    net.eval()
                    new_sample = net.predict(data)[:,1:-1]
#                    print('\n')
#                    print('\n')
#                    print(data)
#                    print(data.shape)
#                    print(target)
#                    print(target.shape)
#                    print(new_sample)
#                    print(new_sample.shape)
                    logits = net(data, target[:,:-1])
                    p = F.softmax(logits, dim = 1)
                    this_sample_p = torch.gather(p, 1, new_sample.unsqueeze(dim=1)).squeeze(dim=1)
                    entropy = -torch.log(this_sample_p).sum(dim=-1)
                    # * fix the start of word token
                    new_sample = new_sample-2
                    new_sample[new_sample<0] = 0

                    # check how many samples conserved the density
                    ok_samples=0
                    for i in range(args.batch_size):
                        if (target[i]==2).sum()==(new_sample[i]==0).sum():
                            if (target[i]==3).sum()==(new_sample[i]==1).sum():
                                if (target[i]==4).sum()==(new_sample[i]==2).sum():
                                    ok_samples+=1

                    # I have to resort the samples to compute energy
                    for s_i in range(args.batch_size):
                        newsample_resorted[s_i] = mysorting(reverse_order, torch.concat([patches[s_i,:98],new_sample[s_i]],dim=-1))
                        oldsample_resorted[s_i] = mysorting(reverse_order, patches[s_i])

                    new_energy = torch.Tensor(np.asarray(latticeglass.patch_energy(newsample_resorted, onedmask.cpu().detach().numpy(), nn_list_patch, args.q)))
                    old_energy = torch.Tensor(np.asarray(latticeglass.patch_energy(oldsample_resorted, onedmask.cpu().detach().numpy(), nn_list_patch, args.q)))
                    zeros = (new_sample == 0).sum()
                    ones = (new_sample == 1).sum()
                    twos = (new_sample == 2).sum()
                    N = args.batch_size*(3**3)
                    my_log('\naverage density of zeros={:.3g}\tones={:.3g}\ttwos={:.3g}\n{}/{} samples conserved the density'.format(zeros/N,ones/N,twos/N,ok_samples,args.batch_size)) 
                    net.train()


            logits = net(data, target[:,:-1])
            loss = criterion(logits, target[:,1:]) #exclude from loss the start of word token

            # and measure the validation loss
            #val_loss = -net.log_prob(val_sample_wrho).mean()
            val_loss = 0*loss

            sample_time += time.time() - sample_start_time
            train_start_time = time.time()


        #  ************
        # Zero the gradient    
        optimizer.zero_grad()
    
        # If required add regularization
        if args.regularization == 'l1':
            l1_norm = sum(p.sum() for p in net.parameters())
            loss_reinforce += args.lambdal1 * l1_norm
        if args.regularization == 'l2':
            l2_norm = sum(p.pow(2.0).sum() for p in net.parameters())
            loss_reinforce += args.lambdal2 * l2_norm

        # Backpropagate
        loss.backward()
    
        if args.clip_grad:
            nn.utils.clip_grad_norm_(params, args.clip_grad)
    
        optimizer.step()
    
        # If I am doing maxlike, I will start scheduler and early stop from epoch=50
        # while if I am doing var, I will start only when beta~equi

        #if args.lr_schedule and args.ARtype == 'maxlike' and step>50 :
        #    scheduler.step(loss.mean())
        #elif args.lr_schedule and round(beta,3) == args.beta:
        #    scheduler.step(loss.mean())
        if args.lr_schedule:
            scheduler.step()

        # Notice that early stopping is done over the validation loss 
        if args.early_stopping and args.ARtype == 'maxlike' and step>50 :
             early_stopping(val_loss)
             if early_stopping.early_stop:
                 my_log('Exiting for early stopping trigger')
                 break
        elif args.early_stopping and round(beta,3) == args.beta:
             early_stopping(val_loss)
             if early_stopping.early_stop:
                 my_log('Exiting for early stopping trigger')
                 break

        train_time += time.time() - train_start_time

        if args.print_step and step % args.print_step == 0:
            entropy_mean = float(entropy.mean())/((args.patch_size-2)**3) 
            energy_mean = float(new_energy.mean())/((args.patch_size-2)**3) 
            old_energy_mean = float(old_energy.mean())/((args.patch_size-2)**3) 
            free_energy = energy_mean - entropy_mean/ args.beta 
            if step > 0:
                sample_time /= args.print_step
                train_time /= args.print_step
            used_time = time.time() - start_time
            my_log(
                'step = {}, F = {:.4g}, S/N = {:.4g}, E(generated)/N = {:.5g}, E(ref)/N = {:.5g}, E(equi)/N = {:.5g}, lr = {:.3g}, beta = {:.3g}, T={:.3g}, sample_time = {:.3f}, loss = {:.3g}, val_loss = {:.4g}, train_time = {:.3f}, used_time = {:.3f}'
                .format(
                    step,
                    free_energy,
                    entropy_mean,
                    energy_mean,
                    old_energy_mean,
                    E_equi,
                    optimizer.param_groups[0]['lr'],
                    beta,
                    1./max(0.000001,beta),
                    sample_time,
                    loss,
                    val_loss,
                    train_time,
                    used_time,
                ))
            sample_time = 0
            train_time = 0


            if args.save_sample:
                state = {
                    'sample': sample,
                    'log_prob': log_prob,
                    'energy': energy,
                    'loss': loss,
                }
                torch.save(state, '{}_save/{}.sample'.format(
                    args.out_filename, step))

        if (args.out_filename and args.save_step
                and step % args.save_step == 0):
            state = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            if args.lr_schedule:
                state['scheduler'] = scheduler.state_dict()
            torch.save(state, '{}_save/{}.state'.format(
                args.out_filename, step))

        if (args.out_filename and args.visual_step
                and step % args.visual_step == 0):

            if args.print_sample and 0:
                energy_np = np.asarray(new_energy)
                energy_count = np.stack(
                    np.unique(energy_np, return_counts=True)).T
                my_log(
                    '\nsample\n{}\nlog_prob\n{}\nenergy\n{}\nloss\n{}\nenergy_count\n{}\n'
                    .format(
                        sample[:args.print_sample, :],
                        log_prob[:args.print_sample],
                        energy_np,
                        loss[:args.print_sample],
                        energy_count,
                    ))

            if args.print_grad:
                my_log('grad max_abs min_abs mean std')
                for name, param in named_params:
                    if param.grad is not None:
                        grad = param.grad
                        grad_abs = torch.abs(grad)
                        my_log('{} {:.3g} {:.3g} {:.3g} {:.3g}'.format(
                            name,
                            torch.max(grad_abs).item(),
                            torch.min(grad_abs).item(),
                            torch.mean(grad).item(),
                            torch.std(grad).item(),
                        ))
                    else:
                        my_log('{} None'.format(name))
                my_log('')


if __name__ == '__main__':
    main()
