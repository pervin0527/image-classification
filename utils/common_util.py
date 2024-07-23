import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Process config path.")
    parser.add_argument('--config_path', type=str, default='./configs/train_base.yaml', help='Path to the config file')
    args = parser.parse_args()

    return args