# Fides Distillation

train_model.py can be used to train a model from scratch to so that service providers can generate a trustworthy pretrained model. Run - 
```bash
	$   python train_model.py [dataset name]
```
architecture can be changed by changing output variable in the script.

distillation_vanilla.py can be used to distill model from scratch. Run - 
```bash
	$   python distillation_vanilla.py [dataset name] [teacher model]
```
Student model architecture can be changed in the code by changing the student variable in the script.

GDTL.py can be used to distill model to pretrained model. Run - 
```bash
	$   python distillation_FT.py [student model] [teacher model] [dataset name]
```
Number of trainable layers can be modified in the code as a parameter of "unfreeze_model(model, layers)" function in the script.
