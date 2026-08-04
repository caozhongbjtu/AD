import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--win_size', type=int, default=512)
    parser.add_argument('--encoding_dim', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)

    parser.add_argument('--data', type=str, default='DLR')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--beta', type=float, default=0.3)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--scales', type=int, default=3)

    parser.add_argument('--log', type=str, default='AD_result')

    return parser.parse_args()

