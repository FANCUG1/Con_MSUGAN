import os
import torch

from config import get_arguments
import functions
from training_generation import train


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_name', help='input image name for training', required=True)
    parser.add_argument('--gpu', type=int, help='which GPU to use', default=0)

    parser.add_argument('--train_mode', default='generation', help='train mode only generations')
    parser.add_argument('--lr_scale', type=float, help='scaling of learning rate for lower stages', default=0.1)
    parser.add_argument('--train_stages', type=int, help='how many stages to use for training', default=5)

    opt = parser.parse_args(['--input_name', './TI_3D.txt'])

    opt = functions.post_config(opt)

    if not os.path.exists(opt.input_name):
        print("Image does not exist: {}".format(opt.input_name))
        print("Please specify a valid image.")
        exit()

    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)

    dir2save = functions.generate_dir2save(opt)

    if os.path.exists(dir2save):
        print('Trained model already exist: {}'.format(dir2save))
        exit()

    try:
        os.makedirs(dir2save)
    except OSError:
        pass

    with open(os.path.join(dir2save, 'parameters.txt'), 'w') as f:
        for o in opt.__dict__:
            f.write("{}\t-\t{}\n".format(o, opt.__dict__[o]))

    print("Training model ({})".format(dir2save))

    train(opt)




