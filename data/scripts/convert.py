"""Convert pretrained weights from PyTorch to TensorFlow-2 Keras.

Example:
    python convert.py --model RegNetX_1_6GF --out_dir ../weights/
"""
import os
import argparse
import torchvision.models
import tensorflow.keras.layers as layers
import numpy as np
import sys
sys.path.append('.')
import regnet     # noqa: E402


def get_torch_model_name(tf_model_name):
    return {
        'RegNetX_400MF': 'regnet_x_400mf',
        'RegNetX_800MF': 'regnet_x_800mf',
        'RegNetX_1_6GF': 'regnet_x_1_6gf',
        'RegNetX_3_2GF': 'regnet_x_3_2gf',
        'RegNetX_8GF':   'regnet_x_8gf',
        'RegNetX_16GF':  'regnet_x_16gf',
        'RegNetX_32GF':  'regnet_x_32gf',

        'RegNetY_400MF': 'regnet_y_400mf',
        'RegNetY_800MF': 'regnet_y_800mf',
        'RegNetY_1_6GF': 'regnet_y_1_6gf',
        'RegNetY_3_2GF': 'regnet_y_3_2gf',
        'RegNetY_8GF':   'regnet_y_8gf',
        'RegNetY_16GF':  'regnet_y_16gf',
        'RegNetY_32GF':  'regnet_y_32gf',
    }[tf_model_name]


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, required=True,
                        help="model name")
    parser.add_argument('--out_dir', type=str, required=True,
                        help="directory to save the converted model")
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    torch_model = getattr(
        torchvision.models,
        get_torch_model_name(args.model)
    )(pretrained=True)
    torch_model.eval()

    state_dict = torch_model.state_dict()
    torch_weights = {}
    for k, v in state_dict.items():
        if len(v.shape) == 0:
            continue

        layer_name = os.path.splitext(k)[0]
        weights = v.cpu().numpy()
        torch_weights.setdefault(layer_name, []).append(weights)

    tf_model = getattr(regnet, args.model)()

    # Transfer
    for layer in tf_model.layers:
        tf_w = layer.get_weights()
        if len(tf_w) == 0:
            continue

        torch_w = torch_weights[layer.name]
        if isinstance(layer, layers.Conv2D):
            torch_w[0] = np.transpose(torch_w[0], [2, 3, 1, 0])
        if isinstance(layer, layers.Dense):
            torch_w[0] = np.transpose(torch_w[0], [1, 0])

        if args.verbose:
            print(layer.name, [w.shape for w in tf_w], [w.shape for w in torch_w])

        layer.set_weights(torch_w)

    os.makedirs(args.out_dir, exist_ok=True)
    filename = os.path.join(args.out_dir, args.model + '.h5')
    print("\nSaving model to", filename)
    tf_model.save(filename)
    print("Conversion completed.")


if __name__ == '__main__':
    main()
