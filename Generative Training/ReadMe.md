# Fides Generative Training

Note - To start using the Generative Training framework, you need to train a verification model and a service model. The distillation folder provides the scripts to do so.

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

example output - 
```bash
156/156 [==============================] - 2696s 17s/step - GACN loss: 7.0034 - Det loss: 0.9591 - Det Acc: 0.8152 - Corrector Loss: 0.2885 - 
val_Corrector acc: 0.9219 - val_Corrector recall: 0.9528 - val_Corrector Prec: 0.9528 - val_Corrector-F1: 0.9528 - 
val_Accuracy: 0.9531 - val_F1Score: 0.9392 - val_Switch_acc: 0.9433 - val_Switch_F1: 0.9400 - val_Avg_acc: 0.9444 - 
val_avg-F1: 0.9411 - val_FGSM_acc: 0.8832 - val_FGSM-F1: 0.8838
```
