import os
import sys
from glob import glob
import numpy as np
import torch
from args import args


if args.dtype == 'float32':
    default_dtype = np.float32
    default_dtype_torch = torch.float32
elif args.dtype == 'float64':
    default_dtype = np.float64
    default_dtype_torch = torch.float64
else:
    raise ValueError('Unknown dtype: {}'.format(args.dtype))

np.seterr(all='raise')
np.seterr(under='warn')
np.set_printoptions(precision=8, linewidth=160)

torch.set_default_dtype(default_dtype_torch)
torch.set_printoptions(precision=8, linewidth=160)
torch.backends.cudnn.benchmark = True

if not args.seed:
    args.seed = np.random.randint(1, 10**8)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
args.device = torch.device('cpu' if args.cuda < 0 else 'cuda:0')

args.out_filename = None


def get_args_features():
    features = '{n}_{graph_seed}_beta{beta:g}'

    if args.ARtype == 'var':
        features += '_var'
    elif args.ARtype == 'nlMC':
        features += '_nlMC{MCsteps}'
    elif args.ARtype == 'maxlike':
        features += '_maxlike'
    else:
        print('Error: ARtype {ARtype} not available') 
        sys.exit()

    if args.net == 'made':
        features += '_nd{net_depth}_nw{net_width}_made'
    elif args.net == 'coloredmade':
        features += '_nd{net_depth}_nw{net_width}_coloredmade'
    elif args.net == 'nade':
        features += '_nd{net_depth}_nw{net_width}_nade'
    elif args.net == 'deepnade':
        features += '_nd{net_depth}_nw{net_width}_deepnade'
    elif args.net == 'gnn':
        features += '_nd{net_depth}_nw{net_width}_gd{graph_depth}_gnn'
    elif args.net == 'coloredgnn':
        features += '_nd{net_depth}_nw{net_width}_gd{graph_depth}_coloredgnn'
    elif args.net == 'transformer':
        features += '_nd{net_depth}_nw{net_width}_nh{nheads}_transformer'
    elif args.net == 'pixelcnn':
        features += '_nd{net_depth}_nw{net_width}_hks{half_kernel_size}'
    else:
        print('Error: network {net} not available') 
        sys.exit()

    if args.bias:
        features += '_bias'
    if args.x_hat_clip:
        features += '_xhc{x_hat_clip:g}'

    if args.optimizer != 'adam':
        features += '_{optimizer}'
    if args.lr_schedule:
        features += '_lrs'
    if args.beta_anneal:
        features += '_ba{beta_anneal:g}'
    if args.res_block:
        features += '_resblock'
    if args.clip_grad:
        features += '_cg{clip_grad:g}'

    if args.regularization:
        features += '_{regularization}'
        if args.regularization == 'l1':
            features += '{lambdal1}'
        elif args.regularization == 'l2':
            features += '{lambdal2}'
        elif args.regularization == 'dropout':
            features += '{dropout_val}'
        else:
            print('Error: regularization {regularization} not available')


    features = features.format(**vars(args))

    return features


def init_out_filename():
    if not args.out_dir:
        return
    features = get_args_features()
    template = '{args.out_dir}/{features}{args.out_infix}/out'
    args.out_filename = template.format(**{**globals(), **locals()})


def ensure_dir(filename):
    dirname = os.path.dirname(filename)
    if dirname:
        try:
            os.makedirs(dirname)
        except OSError:
            pass


def init_out_dir():
    if not args.out_dir:
        return
    init_out_filename()
    ensure_dir(args.out_filename)
    if args.save_step:
        ensure_dir(args.out_filename + '_save/')


def clear_log():
    if args.out_filename:
        open(args.out_filename + '.log', 'w').close()


def clear_err():
    if args.out_filename:
        open(args.out_filename + '.err', 'w').close()


def my_log(s):
    if args.out_filename:
        with open(args.out_filename + '.log', 'a', newline='\n') as f:
            f.write(s + u'\n')
    if not args.no_stdout:
        print(s)


def my_err(s):
    if args.out_filename:
        with open(args.out_filename + '.err', 'a', newline='\n') as f:
            f.write(s + u'\n')
    if not args.no_stdout:
        print(s)


def print_args(print_fn=my_log):
    for k, v in args._get_kwargs():
        if print_fn==my_log:
            print_fn('{} = {}'.format(k, v))
        else:
            print_fn('{} = {}\n'.format(k, v))
    print_fn('')


def parse_checkpoint_name(filename):
    filename = os.path.basename(filename)
    filename = filename.replace('.state', '')
    step = int(filename)
    return step


def get_last_checkpoint_step():
    if not (args.out_filename and args.save_step):
        return -1
    filename_list = glob('{}_save/*.state'.format(args.out_filename))
    if not filename_list:
        return -1
    step = max([parse_checkpoint_name(x) for x in filename_list])
    return step


def clear_checkpoint():
    if not (args.out_filename and args.save_step):
        return
    filename_list = glob('{}_save/*.state'.format(args.out_filename))
    for filename in filename_list:
        os.remove(filename)


# Do not load some params
def ignore_param(state, net):
    ignore_param_name_list = ['x_hat_mask', 'x_hat_bias']
    param_name_list = list(state.keys())
    for x in param_name_list:
        for y in ignore_param_name_list:
            if y in x:
                state[x] = net.state_dict()[x]
                break


def sample_store(sample, ordered_spins, is_ordered=True):
    ensure_dir(args.out_filename + '_samples/')

    # Check if I have to reorder the spins 
    if is_ordered is True:
        # Here the spin are not rearrenged by the AR model
        print('Spins do not need to be ordered')
        for sample_n in range(args.batch_size):
            with open(args.out_filename + '_samples/sample%d_N%d_c%s_s%s_T%.2f.txt'%(sample_n,args.n,args.c,args.graph_seed,1./args.beta), "w+") as text_file:
                for spin in sample[sample_n,:]:
                    text_file.write("%s\n" %int(spin))
    else:
        # I need to put back the spin ordere before storing the sample
        print('Spins need to be ordered')
        # here I create the list to put them back
        ri_ordered_spins=np.zeros(args.n,dtype=int)
        for index, pos in enumerate(ordered_spins):
            ri_ordered_spins[pos] = index 
        for sample_n in range(args.batch_size):
            with open(args.out_filename + '_samples/sample%d_N%d_c%s_s%s_T%.2f.txt'%(sample_n,args.n,args.c,args.graph_seed,1./args.beta), "w+") as text_file:
                for index,spin in enumerate(sample[sample_n,:]):
                    text_file.write("%s\n" %int(sample[sample_n,ri_ordered_spins[index]]))


def gen_log_space(limit, n):
    result = [1]
    if n>1:  # just a check to avoid ZeroDivisionError
        ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    while len(result)<n:
        next_value = result[-1]*ratio
        if next_value - result[-1] >= 1:
            # safe zone. next_value will be a different integer
            result.append(next_value)
        else:
            # problem! same integer. we need to find next_value by artificially incrementing previous value
            result.append(result[-1]+1)
            # recalculate the ratio so that the remaining values will scale correctly
            ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
    return np.array(list(map(lambda x: round(x)-1, result)), dtype=np.uint64)


def moving_average(x,window):
    window=10000
    mean_x = np.zeros(len(x))
    count_x = np.zeros(len(x))
    for i,x_value in enumerate(x):
        mean_x[i:(i+window)]+=x_value
        count_x[i:(i+window)] += 1
    for i,m in enumerate(mean_x):
        mean_x[i]=mean_x[i]/count_x[i]
    return mean_x

#---------------------------------------------------------------
# Functions for the color gradient
def hex_to_RGB(hex):
    ''' "#FFFFFF" -> [255,255,255] '''
    # Pass 16 to the integer function for change of base
    return [int(hex[i:i+2], 16) for i in range(1,6,2)]
def RGB_to_hex(RGB):
    ''' [255,255,255] -> "#FFFFFF" '''
    # Components need to be integers for hex to make sense
    RGB = [int(x) for x in RGB]
    return "#"+"".join(["0{0:x}".format(v) if v < 16 else "{0:x}".format(v) for v in RGB])

def color_dict(gradient):
    # Takes in a list of RGB sub-lists and returns dictionary of
    #colors in RGB and hex form for use in a graphing function
    #defined later on 
    return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
        "r":[RGB[0] for RGB in gradient],
        "g":[RGB[1] for RGB in gradient],
        "b":[RGB[2] for RGB in gradient]}
def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
    # returns a gradient list of (n) colors between
    # two hex colors. start_hex and finish_hex
    # should be the full six-digit color string,
    # inlcuding the number sign ("#FFFFFF")
    # Starting and ending colors in RGB form
    s = hex_to_RGB(start_hex)
    f = hex_to_RGB(finish_hex)
    # Initilize a list of the output colors with the starting color
    RGB_list = [s]
    # Calcuate a color at each evenly spaced value of t from 1 to n
    for t in range(1, n):
        # Interpolate RGB vector for color at the current value of t
        curr_vector = [
          int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
          for j in range(3)
        ]
        # Add it to our list of output colors
        RGB_list.append(curr_vector)
    
    return color_dict(RGB_list)
#---------------------------------------------------------------


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
#            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


def mysorting(order,content):
    sorted_content = content*0
    if len(order)!=len(content):
        print('Error! can not sort! {} != {}'.format(len(order),len(content)))
        print(order)
        print(content)
        sys.exit()
    for i in range(len(order)):
        sorted_content[i] = content[order[i]]
    return sorted_content
