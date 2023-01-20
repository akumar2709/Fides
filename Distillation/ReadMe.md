# Fides Distillation
distillation_vanilla.py can be used to distill model from scratch. Run - 
```bash
	$   python distillation_vanilla.py [dataset name] [teacher model]
```
Student model architecture can be changed in the code by changing the student variable.

GDTL.py can be used to distill model to pretrained model. Run - 
```bash
	$   python distillation_vanilla.py [student model] [teacher model] [dataset name]
```
Number of trainable layers can be modified in the code as a parameter of "unfreeze_model(model, layers)" function.
