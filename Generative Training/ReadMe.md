# Fides Generative Training

GACN_cifar.py can be used train the detector on cifar10 or cifar100. Run - 
```bash
	$   python GACN_cifar.py [dataset name] [verifiation model] [service model]
```
example - 
```bash
	$   python GACN_cifar.py cifar10 ResNet50.h5 ResNet152.h5
```

GACN_imagenet.py can be used to train the detector and corrector on ImageNet dataset. Run - 
```bash
	$   python GACN_imagenet.py [verifiation model] [service model]
```
example - 
```bash
	$   python GACN_imagenent.py ResNet50.h5 ResNet152.h5
```
