import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_root_dir', '-crd',
                    default='/media/data_1/snva/Checkpoints/fhwa')
parser.add_argument('--dataset_name', '-dn', required=True)

args = parser.parse_args()

ckpt_dir = os.path.join(args.checkpoint_root_dir, args.dataset_name)

if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)

archs = ['densenet_bc_12_100', 'densenet_bc_24_50', 'inception_resnet_v2',
         'inception_v3', 'inception_v4', 'mobilenet_v1']

inits = ['random_init', 'transfer_init']

vals = ['eval', 'test']

for arch in archs:
    arch_dir = os.path.join(ckpt_dir, arch)

    if not os.path.exists(arch_dir):
        os.mkdir(arch_dir)

    for init in inits:
        init_dir = os.path.join(arch_dir, init)

        if not os.path.exists(init_dir):
            os.mkdir(init_dir)

        for val in vals:
            val_dir = os.path.join(init_dir, val)

            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
