
import argparse


def args_n():
    parser = argparse.ArgumentParser(description="NPDE")

    # 显示设置
    parser.add_argument('--vis_name', default='mamba_combine', type=str, help='visdom name')

    # 数据设置
    parser.add_argument('--trainh5',
                       default='/home/mzq/file_soft/data/new/L_1/train.h5', type=str, help='Path of the training data')
    parser.add_argument('--test_data',
                        default='/home/mzq/file_soft/data/texture_256/test/', type=str, help='Path of test dataset')
    # parser.add_argument('--test_data',
    #                     default='test_data/', type=str, help='Path of test dataset')
    parser.add_argument('--valh5',
                        default='/home/mzq/file_soft/data/Model/more/L_1/valid.h5', help='Path of the validation data')
    parser.add_argument('--patch_size', default=128, type=int, help='Patch size')
    # parser.add_argument('--patch_per_image', default=100, type=int, help='Number of patches per image')
    parser.add_argument('--batch_size', default=128, type=int, help='batch Batch size')

    parser.add_argument('--L', type=float, default=1., help='Noise level for Gaussian noise')
    parser.add_argument('--is_blind', action='store_true', help='When train the model, blind noise or not')

    # 结果保存
    parser.add_argument('--save_models', default='Test_time',  help='Save trained models')
    parser.add_argument('--save_path', default='Test_time', help='Save output results')

    # 网络训练设置
    parser.add_argument('--optimizer', default='ADAM',  choices=('SGD', 'ADAM', 'RMSprop'),
                        help='Optimizer to use (SGD | ADAM | RMSprop)')
    parser.add_argument('--loss', type=str, default = '1*MSE', help='Loss function configuration')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='Beta values for ADAM optimizer')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Epsilon value for ADAM optimizer')
    parser.add_argument('--print_every', type=int, default=100,
                        help='Number of batches to wait before logging the training status')

    parser.add_argument('--epochs', type=int, default=20, help='The largest number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_patience', default=3, type=int, help='Number of epochs to wait for learning rate decay')
    parser.add_argument('--decay_rate', default=0.5, type=float, help='Decay rate for learning rate')

    parser.add_argument('--patience', type=int, default=50, help='Number of steps for early stopping')
    parser.add_argument('--num_channel', type=int, default=1,
                        help='Number of image channels. Use 1 for grayscale and 3 for color')
    parser.add_argument('--out_channel', type=int, default=128,
                        help='Number of image channels. Use 1 for grayscale and 3 for color')
    parser.add_argument('--denoise_num', type=int, default=1,
                        help='Number of image channels. Use 1 for grayscale and 3 for color')


    # 设备设置
    parser.add_argument('--n_GPUs', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--cudanum', type=str, default='1', help='Ordinal number of the CUDA device to be used')
    parser.add_argument('--random_seed', type=int, default=1200, help='Random seed to use for generating random numbers')
    parser.add_argument('--device', type=str, default='cuda', help='Device to be used for computation')

    parser.add_argument('--model_name', default='SAMNDE', type=str, help='Type of model to use')
    parser.add_argument('--odeint_adjoint', default=False, help='For NODE, use odeint_adjoint or not')
    parser.add_argument('--N_t', type=int, default=4, help='Number of time steps')
    parser.add_argument('--ODE_vector_field', default='_ODEVectorField128', type=str, help='Vector field model to use')
    parser.add_argument('--solver', type=str, default = 'f_random_euler', help='NODE solver to use')
    parser.add_argument('--t', type=float, default=1, help='Value of T')
    parser.add_argument('--random_euler_random_seed', default=8888, help='Random seed for f random euler')

    parser.add_argument('--test_only', default=False, type=bool, help='Perform testing only or not')
    parser.add_argument('--load_model', default=False, help='Load a pretrained model or not')
    parser.add_argument('--load_model_path', default='./snapshot/save_CNN_XNODE_X_sigma50.0_2022-12-09-01-51-51/experiment/test/model/model_best.pt', help='Path of the pretrained model')


    args = parser.parse_args()
    return args