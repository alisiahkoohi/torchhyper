import argparse
import json
import hashlib


def shorten_filename(filename, max_length=255):
    """Shorten the filename using a hash if it exceeds max_length."""
    if len(filename) > max_length:
        hash_object = hashlib.sha1(filename.encode())
        hash_filename = hash_object.hexdigest()
        extension = filename.split('.')[-1] if '.' in filename else ''
        short_filename = (f"{filename[:max_length - len(hash_filename) - 1]}"
                          f".{hash_filename}" if extension else hash_filename)
        return short_filename
    return filename


def read_config(filename):
    """Read input variables and values from a json file."""
    with open(filename) as f:
        configs = json.load(f)
    return configs


def write_config(args, filename):
    "Write command line arguments into a json file."
    with open(filename, 'w') as f:
        json.dump(args, f)


def parse_input_args(args):
    "Use variables in args to create command line input parser."
    parser = argparse.ArgumentParser(description='')
    for key, value in args.items():
        parser.add_argument('--' + key, default=value, type=type(value))
    return parser.parse_args()


def make_experiment_name(args):
    """Make experiment name based on input arguments"""
    experiment_name = args.experiment_name + '_'
    for key, value in vars(args).items():
        if key not in [
                'experiment_name',
                'gpu_id',
                'phase',
                'val_batchsize',
                'testing_epoch',
                'time_emb',
                'beta_schedule',
                'num_val',
                'emb_size',
                'save_freq',
                'upload_results',
        ]:
            experiment_name += key + '-{}_'.format(value)
    experiment_name = experiment_name[:-1].replace(' ', '').replace(',', '-')
    return shorten_filename(experiment_name)


def process_sequence_arguments(args):
    """Process sequence arguments to remove spaces and split by comma."""
    if hasattr(args, 'hidden_size'):
        args.hidden_size = args.hidden_size.replace(' ', '').split(',')
        args.hidden_size = [int(_) for _ in args.hidden_size]
    if hasattr(args, 'block_channels'):
        args.block_channels = args.block_channels.replace(' ', '').split(',')
        args.block_channels = [int(_) for _ in args.block_channels]
    if hasattr(args, 'mix_weights'):
        # This argument is a string of comma-separated floats, e.g.,
        # "1.0,1.0"1,0".
        args.mix_weights = [
            float(_) for _ in args.mix_weights.replace(' ', '').split(',')
        ]
    if hasattr(args, 'mix_means'):
        # This argument is a string of comma-separated tuples, e.g.,
        # "(0.0,0.0),(1.0,1.0)". If it is a string of comma-separated floats,
        # create a list of lists with args.input_size elements each obtained by
        # repeating the same float.
        if '(' not in args.mix_means:
            args.mix_means = [
                list(float(pair) for _ in range(args.input_size))
                for pair in args.mix_means.replace(' ', '').split(',')
            ]
        else:
            args.mix_means = [
                list(map(float, pair.split(',')))
                for pair in args.mix_means.strip('()').split('),(')
            ]
    if hasattr(args, 'mix_vars'):
        # This argument is a string of comma-separated tuples, e.g.,
        # "(0.0,0.0),(1.0,1.0)". If it is a string of comma-separated floats,
        # create a list of lists with args.input_size elements each obtained by
        # repeating the same float.
        if '(' not in args.mix_vars:
            args.mix_vars = [
                list(float(pair) for _ in range(args.input_size))
                for pair in args.mix_vars.replace(' ', '').split(',')
            ]
        else:
            args.mix_vars = [
                list(map(float, pair.split(',')))
                for pair in args.mix_vars.strip('()').split('),(')
            ]
    # Check if the number of elements in mix_weights, mix_means, and mix_vars
    # is the same.
    if (hasattr(args, 'mix_weights') and hasattr(args, 'mix_means')
            and hasattr(args, 'mix_vars')):
        # Raise an error if the number of elements in mix_weights, mix_means,
        # and mix_vars is not the same.
        if len(args.mix_weights) != len(args.mix_means) or len(
                args.mix_weights) != len(args.mix_vars):
            raise ValueError(
                'The number of elements in mix_weights, mix_means, and '
                'mix_vars must be the same.')
    if hasattr(args, 'subclass'):
        args.subclass = args.subclass.replace(' ', '').split(',')
        args.subclass = [int(_) for _ in args.subclass]
    if hasattr(args, 'data_perc'):
        if isinstance(args.data_perc, str):
            args.data_perc = args.data_perc.replace(' ', '').split(',')
            args.data_perc = [float(_) for _ in args.data_perc]
    return args
