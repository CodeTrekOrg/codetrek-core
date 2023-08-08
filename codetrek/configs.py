import argparse

cmd_opt = argparse.ArgumentParser(description='Argparser', allow_abbrev=False)

cmd_opt.add_argument('--data_dir', default='tmp/data', help='data root directory')
cmd_opt.add_argument('--dict', default='dict.pkl', help='name of the dictionary file')
cmd_opt.add_argument('--seed', default=19260817, type=int, help='seed')
cmd_opt.add_argument('--embed_dim', default=256, type=int, help='embed size')
cmd_opt.add_argument('--nhead', default=4, type=int, help='multi-head attention')
cmd_opt.add_argument('--dropout', default=0, type=float, help='dropout')
cmd_opt.add_argument('--phase', default='train', help='train/test')
cmd_opt.add_argument('--model_dump', default='model-best_dev.ckpt', help='load model dump')
cmd_opt.add_argument('--device', type=int, default=-1, help='-1: cpu; 0 - ?: specific gpu index')
cmd_opt.add_argument('--grad_clip', default=5, type=float, help='gradient clip')
cmd_opt.add_argument('--transformer_layers', default=9, type=int, help='# transformer layers')
cmd_opt.add_argument('--dim_feedforward', default=512, type=int, help='embed size')
cmd_opt.add_argument('--num_epochs', default=1000, type=int, help='num epochs')
cmd_opt.add_argument('--batch_size', default=16, type=int, help='batch size')
cmd_opt.add_argument('--learning_rate', default=1e-6, type=float, help='learning rate')
cmd_opt.add_argument('--iter_per_epoch', default=300, type=int, help='num iterations per epoch')
cmd_opt.add_argument('--num_walks', default=100, type=int, help='number of random walks per file')
cmd_opt.add_argument('--num_steps', default=10, type=int, help='number of steps in each random walks=16')

cmd_args, _ = cmd_opt.parse_known_args()