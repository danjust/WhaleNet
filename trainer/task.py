import argparse
import pandas as pd

from trainer import model
from trainer import datatools

def train_model():
    parser = argparse.ArgumentParser()

    # Training arguments
    parser.add_argument(
        '--batch_size',
        help = 'Batch size',
        default = 32,
        type = int
    )
    parser.add_argument(
        '--train_steps',
        help = 'Number of steps to train',
        default = 5000,
        type = int
    )
    parser.add_argument(
        '--print_step',
        help = 'output/summary after this number of steps',
        default = 500,
        type = int
    )
    parser.add_argument(
        '--data_dir',
        help = 'Directory with training data',
        default = 'gs://whaleimgs/data/train',
        type = str
    )
    parser.add_argument(
        '--labels_file',
        help = 'csv file with image labels',
        default = 'gs://whaleimgs/data/labels_c',
        type = str
    )
    parser.add_argument(
        '--learning_rate',
        help = 'learning rate',
        default = 0.001,
        type = float
    )
    parser.add_argument(
        '--margin',
        help = 'margin for contrastive loss',
        default = 1,
        type = int
    )
    args = parser.parse_args()

    # Read labels and create data pipeline
    labels = pd.read_csv(args.labels_file)
    pipeline = datatools.datapipeline(
        datadir = args.data_dir,
        labels = labels,
        batchsize = args.batch_size)
    dataset = pipeline.build()
    iter = dataset.make_initializable_iterator()

    # Create model and train
    net = model.whalenet(
        iter,
        learning_rate = args.learning_rate,
        margin = args.margin,
        model_name = "whale1")

    net.train(
        train_steps = args.train_steps,
        print_step = args.print_step)


if __name__ == '__main__':
    train_model()
