import argparse

cmd_opt = argparse.ArgumentParser(description='Argparser', allow_abbrev=False)

cmd_opt.add_argument('--data_dir', default='tmp/data', help='data root directory')
cmd_opt.add_argument('--output_dir', default='tmp', help='data output directory')
cmd_opt.add_argument('--model_dir', default='tmp/model', help='model root directory')
cmd_opt.add_argument('--split', default='80:10', help='train:dev split percentage of the samples. test percentage will be 100-train-dev.')
cmd_opt.add_argument('--phase', default='train', help='train/test/predict')
cmd_opt.add_argument('--device', type=int, default=-1, help='-1: cpu; 0: cuda/mps')

cmd_args, _ = cmd_opt.parse_known_args()
