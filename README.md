# TensorFlow-2 Pretrained RegNets
RegNet ([Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)) implementation in TensorFlow-2
with pretrained weights.

## Dependencies
* Python ≥ 3.7
* TensorFlow ≥ 2.5

## Models
### RegNetX
| Model name     | Model name in paper | Pretrained weights                                                                                     | Top-1 (%) |
|----------------|---------------------|--------------------------------------------------------------------------------------------------------|:---------:|
| RegNetX_400MF  | REGNETX-400MF       | [RegNetX_400MF.h5](https://drive.google.com/file/d/1vS18w4k6Y62zunhprFzzEqdiEISFn8YZ/view?usp=sharing) | 72.87     |
| RegNetX_800MF  | REGNETX-800MF       | [RegNetX_800MF.h5](https://drive.google.com/file/d/1JUrOurjbbyOH3ezVghf_SGKIQMVg4Gs4/view?usp=sharing) | 75.21     |
| RegNetX_1_6GF  | REGNETX-1.6GF       | [RegNetX_1_6GF.h5](https://drive.google.com/file/d/1WwTA8JUh8TU2k9m7q6tT-SB08ffwRcpO/view?usp=sharing) | 77.11     |
| RegNetX_3_2GF  | REGNETX-3.2GF       | [RegNetX_3_2GF.h5](https://drive.google.com/file/d/1gCXhV4Fck-KEGucyQAUi9H608MmCBNhA/view?usp=sharing) | 78.33     |
| RegNetX_8GF    | REGNETX-8.0GF       | [RegNetX_8GF.h5](https://drive.google.com/file/d/1z08WpI6sB8Trtx4FNte9OXPKcobVEbIC/view?usp=sharing)   | 79.36     |
| RegNetX_16GF   | REGNETX-16GF        | [RegNetX_16GF.h5](https://drive.google.com/file/d/1ulPdlZaXOc6I2uM5GJcqM3UwWJ5p1d5x/view?usp=sharing)  | 79.98     |
| RegNetX_32GF   | REGNETX-32GF        | [RegNetX_32GF.h5](https://drive.google.com/file/d/1EexVfYAYF7ode_LQky1onsnSHN8Nmrru/view?usp=sharing)  | 80.58     |

### RegNetY
| Model name     | Model name in paper | Pretrained weights                                                                                     | Top-1 (%) |
|----------------|---------------------|--------------------------------------------------------------------------------------------------------|:---------:|
| RegNetY_400MF  | REGNETY-400MF       | [RegNetY_400MF.h5](https://drive.google.com/file/d/1QIQhrwplNU8tn-X0IlKqSJQk3nb9RwK_/view?usp=sharing) | 74.02     |
| RegNetY_800MF  | REGNETY-800MF       | [RegNetY_800MF.h5](https://drive.google.com/file/d/1PoWshP02vZQh7P4olZa2H7gzmZCRaich/view?usp=sharing) | 76.44     |
| RegNetY_1_6GF  | REGNETY-1.6GF       | [RegNetY_1_6GF.h5](https://drive.google.com/file/d/102gKOI47ZUWEB8HPfaX9jimjgRC1jrrP/view?usp=sharing) | 77.98     |
| RegNetY_3_2GF  | REGNETY-3.2GF       | [RegNetY_3_2GF.h5](https://drive.google.com/file/d/1OAyOK9B084RO76CdXRjE-XWUPMhdVnb1/view?usp=sharing) | 78.94     |
| RegNetY_8GF    | REGNETY-8.0GF       | [RegNetY_8GF.h5](https://drive.google.com/file/d/1RFx7gL1_X7jppn4Vi1MOjdM4QqRcdtlA/view?usp=sharing)   | 80.05     |
| RegNetY_16GF   | REGNETY-16GF        | [RegNetY_16GF.h5](https://drive.google.com/file/d/1SrYe1UEmm2V_X6AS0_xfac1b7wGpXLNf/view?usp=sharing)  | 80.43     |
| RegNetY_32GF   | REGNETY-32GF        | [RegNetY_32GF.h5](https://drive.google.com/file/d/1Zyfnuvm8RP9wmXvdB0Gj_POJX-_ahjNE/view?usp=sharing)  | 80.84     |

* Pretrained weights are converted from [TorchVision model zoo](https://pytorch.org/vision/stable/models.html#classification),
and we only provide models of certain flop regimes that are available in the model zoo.
Script for conversion is [`data/scripts/convert.py`](data/scripts/convert.py).
* Top-1: 224x224 single-crop, top-1 accuracy using converted weights.
Reproduce by:
```bash
# You need to register on http://www.image-net.org/download-images to get the link to
# download ILSVRC2012_img_val.tar.
mkdir ILSVRC2012_img_val/
tar xvf ILSVRC2012_img_val.tar -C ILSVRC2012_img_val/

python data/scripts/eval.py --h5 path/to/pretrained.h5 --data_dir ILSVRC2012_img_val/ --batch_size 32
```

## Usage
```python
import regnet

# Specify include_top=False if you want to remove the classification layer at the top
model = regnet.RegNetX_1_6GF(input_shape=(224, 224, 3),
                             weights="path/to/RegNetX_1_6GF.h5",
                             include_top=True)

model.compile(...)
model.fit(...)
```
**Note**: Input images should be loaded in to a range of [0, 1] and then normalized using
mean = `[0.485, 0.456, 0.406]` and std = `[0.229, 0.224, 0.225]`.

## License
MIT
