import argparse

parser = argparse.ArgumentParser()

group = parser.add_argument_group('physics parameters')
group.add_argument(
    '--n',
    type=int,
    help='number of spins')
group.add_argument(
    '--q',
    type=int,
    default=10,
    help='number of colors')
group.add_argument(
    '--c',
    type=int,
    default=40,
    help='connectivity of the graph')
group.add_argument(
    '--graph_seed',
    type=int,
    help='seed from which I generated the graph')

group.add_argument('--beta', type=float, default=1, help='beta = 1 / k_B T')
group.add_argument('--T', type=float, default=1, help='k_B T')

group = parser.add_argument_group('network parameters')
group.add_argument(
    '--ARtype',
    type=str,
    default='var',
    choices=['var', 'nlMC','maxlike'],
    help='Autoregressive approach type')
group.add_argument(
    '--net',
    type=str,
    default='nade',
    choices=['made', 'nade', 'deepnade', 'gnn', 'coloredmade','coloredgnn', 'transformer','pixelcnn'],
    help='network type')
group.add_argument('--net_depth', type=int, default=3, help='network depth')
group.add_argument('--net_width', type=int, default=64, help='network width')
group.add_argument('--graph_depth', type=int, default=3, help='depth of the graph message passing part')
group.add_argument('--nheads', type=int, default=2, help='number of heads for multi-head attention')
group.add_argument('--hid_width', type=int, default=150, help='width of deepnade hidden layers')
group.add_argument(
    '--half_kernel_size', type=int, default=1, help='(kernel_size - 1) // 2')
group.add_argument('--res_block', action='store_true', help='use res block')
group.add_argument(
    '--final_conv',
    action='store_true',
    help='add an additional conv layer before sigmoid')
group.add_argument(
    '--regularization',
    type=str,
    choices=['l1', 'l2','dropout'],
    help='Which regularization to use')
group.add_argument('--lambdal1', type=float, default=0.000001, help='weight of l1 loss')
group.add_argument('--lambdal2', type=float, default=0.000001, help='weight of l2 loss')
group.add_argument('--dropout_val', type=float, default=0.1, help='percentage of dropout')
group.add_argument(
    '--dtype',
    type=str,
    default='float32',
    choices=['float32', 'float64'],
    help='dtype')
group.add_argument('--bias', action='store_true', help='use bias')
group.add_argument('--acceptance', action='store_true', help='measure the acceptance by sampling from this model')
group.add_argument(
    '--qsym', action='store_true', help='order the sites in order to maximize the shared information')
group.add_argument(
    '--site_ordering', action='store_true', help='use colors permutation symmetry in sample and loss')
group.add_argument(
    '--sample_permute', action='store_true', help='generate samples includin colors permutations')
group.add_argument(
    '--x_hat_clip',
    type=float,
    default=0,
    help='value to clip x_hat around 0 and 1, 0 for disabled')
group.add_argument(
    '--epsilon',
    type=float,
    default=1e-7,
    help='small number to avoid 0 in division and log')
group.add_argument(
    '--M',
    type=int,
    default=1000,
    help='Number of samples to generate to evaluate Meffective')
group.add_argument(
    '--MCsteps',
    type=int,
    default=25000,
    help='MC steps to run before backpropagation')

group = parser.add_argument_group('optimizer parameters')
group.add_argument(
    '--seed', type=int, default=0, help='random seed, 0 for randomized')
group.add_argument(
    '--optimizer',
    type=str,
    default='adam',
    choices=['sgd', 'sgdm', 'rmsprop', 'adam', 'adam0.5'],
    help='optimizer')
group.add_argument(
    '--batch_size', type=int, default=10**3, help='number of samples (or mini-batch)')
group.add_argument(
    '--sample_size', type=int, default=10**3, help='number of samples to generate for maxlike (it should be larger than batch_size)')
group.add_argument(
    '--val_sample_size', type=int, default=10**3, help='number of validation samples to generate')
group.add_argument(
    '--samples_for_autocorr', type=int, default=10**3, help='number of samples to measure autocorrelation')
group.add_argument(
    '--globMC',
    type=str,
    choices=['train', 'AR', 'val'],
    help='which samples to use to initialize globalMC')
group.add_argument('--lr', type=float, default=1e-3, help='learning rate')
group.add_argument(
    '--max_step', type=int, default=10**6, help='maximum number of steps')
group.add_argument(
    '--lr_schedule', action='store_true', help='use learning rate scheduling')
group.add_argument(
    '--early_stopping', action='store_true', help='use early stopping')
group.add_argument(
    '--beta_anneal',
    type=float,
    default=0,
    help='speed to change beta from 0 to final value, 0 for disabled')
group.add_argument(
    '--clip_grad',
    type=float,
    default=0,
    help='global norm to clip gradients, 0 for disabled')

group = parser.add_argument_group('system parameters')
group.add_argument(
    '--no_stdout',
    action='store_true',
    help='do not print log to stdout, for better performance')
group.add_argument(
    '--clear_checkpoint', action='store_true', help='clear checkpoint')
group.add_argument(
    '--print_step',
    type=int,
    default=1,
    help='number of steps to print log, 0 for disabled')
group.add_argument(
    '--save_step',
    type=int,
    default=500,
    help='number of steps to save network weights, 0 for disabled')
group.add_argument(
    '--visual_step',
    type=int,
    default=100,
    help='number of steps to visualize samples, 0 for disabled')
group.add_argument(
    '--save_sample', action='store_true', help='save samples on print_step')
group.add_argument(
    '--print_sample',
    type=int,
    default=1,
    help='number of samples to print to log on visual_step, 0 for disabled')
group.add_argument(
    '--print_grad',
    action='store_true',
    help='print summary of gradients for each parameter on visual_step')
group.add_argument(
    '--cuda', type=int, default=-1, help='ID of GPU to use, -1 for disabled')
group.add_argument(
    '--out_infix',
    type=str,
    default='',
    help='infix in output filename to distinguish repeated runs')
group.add_argument(
    '-o',
    '--out_dir',
    type=str,
    default='out',
    help='directory prefix for output, empty for disabled')

group.add_argument(
    '--start_from', type=int, default=0, help='for the globalMCMC you start processing samples from this id (to avoid conflicts)')

args = parser.parse_args()
