import glob
import sys
import os

list_model=glob.glob('out/[!an]*')
for model in list_model:
    print('\n******\n{}'.format(model))
    with open('{}/out.myargs'.format(model),'w') as fout, open('{}/out.log'.format(model),'r') as fin:
        for line in fin:
            x = line.strip('\n')
            if x == '':
                print('## Exit')
                break
            else:
                print(x)
                fout.write(x+'\n')
