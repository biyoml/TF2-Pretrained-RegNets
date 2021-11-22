import tensorflow as tf
from tensorflow.keras import layers


def _conv(x, name, width, kernel_size, stride=1, groups=1, relu=True):
    x = layers.Conv2D(width,
                      kernel_size=kernel_size,
                      strides=stride,
                      groups=groups,
                      use_bias=False,
                      name=name + '.0')(x)
    x = layers.BatchNormalization(epsilon=1e-5, name=name + '.1')(x)
    if relu:
        x = layers.ReLU(name=name + '.relu')(x)
    return x


def _se(xin, name, width):
    x = layers.GlobalAveragePooling2D(name=name + '.squeeze')(xin)
    x = layers.Reshape([1, 1, -1], name=name + '.reshape')(x)
    x = layers.Conv2D(width,
                      kernel_size=1,
                      activation='relu',
                      name=name + '.fc1')(x)
    x = layers.Conv2D(xin.shape[-1],
                      kernel_size=1,
                      activation='sigmoid',
                      name=name + '.fc2')(x)
    return layers.Multiply(name=name + '.excite')([xin, x])


def _residual_block(xin, name, width, bottleneck_ratio, group_width, stride,
                    se_ratio):
    w_b = round(width * bottleneck_ratio)
    x = _conv(xin,
              name=name + '.f.a',
              width=w_b,
              kernel_size=1)
    x = layers.ZeroPadding2D(1, name=name + '.pad')(x)
    x = _conv(x,
              name=name + '.f.b',
              width=w_b,
              kernel_size=3,
              stride=stride,
              groups=w_b // group_width)

    if se_ratio is not None:
        x = _se(x, name=name + '.f.se', width=round(se_ratio * xin.shape[-1]))

    x = _conv(x,
              name=name + '.f.c',
              width=width,
              kernel_size=1,
              relu=False)

    if (xin.shape[-1] != width) or (stride != 1):
        shortcut = _conv(xin,
                         name=name + '.proj',
                         width=width,
                         kernel_size=1,
                         stride=stride,
                         relu=False)
    else:
        shortcut = xin
    x = layers.Add(name=name + '.add')([shortcut, x])
    return layers.ReLU(name=name + '.relu')(x)


def _stage(x, i, depth, width, bottleneck_ratio, group_width, se_ratio):
    for j in range(depth):
        x = _residual_block(
            x,
            name='trunk_output.block%d.block%d-%d' % (i, i, j),
            width=width,
            bottleneck_ratio=bottleneck_ratio,
            group_width=group_width,
            stride=2 if j == 0 else 1,
            se_ratio=se_ratio,
        )
    return x


def _regnet(
    depths,
    widths,
    bottleneck_ratio,
    group_width,
    se_ratio=None,
    input_shape=(224, 224, 3),
    weights=None,
    include_top=True,
    classes=1000,
):
    input_tensor = layers.Input(input_shape)
    x = layers.ZeroPadding2D(1, name='stem.pad')(input_tensor)
    x = _conv(x, name='stem', width=32, kernel_size=3, stride=2)

    for i, (d, w) in enumerate(zip(depths, widths)):
        x = _stage(x, i + 1, depth=d, width=w, bottleneck_ratio=bottleneck_ratio,
                   group_width=group_width, se_ratio=se_ratio)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='fc')(x)

    model = tf.keras.Model(input_tensor, x)

    if weights is not None:
        model.load_weights(weights, by_name=True)
    return model


def RegNetX_400MF(
    input_shape=(224, 224, 3),
    weights=None,
    include_top=True,
    classes=1000
):
    return _regnet(
        depths=[1, 2, 7, 12],
        widths=[32, 64, 160, 400],
        bottleneck_ratio=1,
        group_width=16,
        input_shape=input_shape,
        weights=weights,
        include_top=include_top,
        classes=classes,
    )


def RegNetX_800MF(
    input_shape=(224, 224, 3),
    weights=None,
    include_top=True,
    classes=1000
):
    return _regnet(
        depths=[1, 3, 7, 5],
        widths=[64, 128, 288, 672],
        bottleneck_ratio=1,
        group_width=16,
        input_shape=input_shape,
        weights=weights,
        include_top=include_top,
        classes=classes,
    )


def RegNetX_1_6GF(
    input_shape=(224, 224, 3),
    weights=None,
    include_top=True,
    classes=1000
):
    return _regnet(
        depths=[2, 4, 10, 2],
        widths=[72, 168, 408, 912],
        bottleneck_ratio=1,
        group_width=24,
        input_shape=input_shape,
        weights=weights,
        include_top=include_top,
        classes=classes,
    )


def RegNetX_3_2GF(
    input_shape=(224, 224, 3),
    weights=None,
    include_top=True,
    classes=1000
):
    return _regnet(
        depths=[2, 6, 15, 2],
        widths=[96, 192, 432, 1008],
        bottleneck_ratio=1,
        group_width=48,
        input_shape=input_shape,
        weights=weights,
        include_top=include_top,
        classes=classes,
    )


def RegNetX_8GF(
    input_shape=(224, 224, 3),
    weights=None,
    include_top=True,
    classes=1000
):
    return _regnet(
        depths=[2, 5, 15, 1],
        widths=[80, 240, 720, 1920],
        bottleneck_ratio=1,
        group_width=120,
        input_shape=input_shape,
        weights=weights,
        include_top=include_top,
        classes=classes,
    )


def RegNetX_16GF(
    input_shape=(224, 224, 3),
    weights=None,
    include_top=True,
    classes=1000
):
    return _regnet(
        depths=[2, 6, 13, 1],
        widths=[256, 512, 896, 2048],
        bottleneck_ratio=1,
        group_width=128,
        input_shape=input_shape,
        weights=weights,
        include_top=include_top,
        classes=classes,
    )


def RegNetX_32GF(
    input_shape=(224, 224, 3),
    weights=None,
    include_top=True,
    classes=1000
):
    return _regnet(
        depths=[2, 7, 13, 1],
        widths=[336, 672, 1344, 2520],
        bottleneck_ratio=1,
        group_width=168,
        input_shape=input_shape,
        weights=weights,
        include_top=include_top,
        classes=classes,
    )


def RegNetY_400MF(
    input_shape=(224, 224, 3),
    weights=None,
    include_top=True,
    classes=1000
):
    return _regnet(
        depths=[1, 3, 6, 6],
        widths=[48, 104, 208, 440],
        bottleneck_ratio=1,
        group_width=8,
        se_ratio=0.25,
        input_shape=input_shape,
        weights=weights,
        include_top=include_top,
        classes=classes,
    )


def RegNetY_800MF(
    input_shape=(224, 224, 3),
    weights=None,
    include_top=True,
    classes=1000
):
    return _regnet(
        depths=[1, 3, 8, 2],
        widths=[64, 144, 320, 784],
        bottleneck_ratio=1,
        group_width=16,
        se_ratio=0.25,
        input_shape=input_shape,
        weights=weights,
        include_top=include_top,
        classes=classes,
    )


def RegNetY_1_6GF(
    input_shape=(224, 224, 3),
    weights=None,
    include_top=True,
    classes=1000
):
    return _regnet(
        depths=[2, 6, 17, 2],
        widths=[48, 120, 336, 888],
        bottleneck_ratio=1,
        group_width=24,
        se_ratio=0.25,
        input_shape=input_shape,
        weights=weights,
        include_top=include_top,
        classes=classes,
    )


def RegNetY_3_2GF(
    input_shape=(224, 224, 3),
    weights=None,
    include_top=True,
    classes=1000
):
    return _regnet(
        depths=[2, 5, 13, 1],
        widths=[72, 216, 576, 1512],
        bottleneck_ratio=1,
        group_width=24,
        se_ratio=0.25,
        input_shape=input_shape,
        weights=weights,
        include_top=include_top,
        classes=classes,
    )


def RegNetY_8GF(
    input_shape=(224, 224, 3),
    weights=None,
    include_top=True,
    classes=1000
):
    return _regnet(
        depths=[2, 4, 10, 1],
        widths=[224, 448, 896, 2016],
        bottleneck_ratio=1,
        group_width=56,
        se_ratio=0.25,
        input_shape=input_shape,
        weights=weights,
        include_top=include_top,
        classes=classes,
    )


def RegNetY_16GF(
    input_shape=(224, 224, 3),
    weights=None,
    include_top=True,
    classes=1000
):
    return _regnet(
        depths=[2, 4, 11, 1],
        widths=[224, 448, 1232, 3024],
        bottleneck_ratio=1,
        group_width=112,
        se_ratio=0.25,
        input_shape=input_shape,
        weights=weights,
        include_top=include_top,
        classes=classes,
    )


def RegNetY_32GF(
    input_shape=(224, 224, 3),
    weights=None,
    include_top=True,
    classes=1000
):
    return _regnet(
        depths=[2, 5, 12, 1],
        widths=[232, 696, 1392, 3712],
        bottleneck_ratio=1,
        group_width=232,
        se_ratio=0.25,
        input_shape=input_shape,
        weights=weights,
        include_top=include_top,
        classes=classes,
    )
