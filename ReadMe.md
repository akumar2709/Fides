# Fides

This is the codebase for Fides: A Generative Framework for Result Validation of Outsourced Machine Learning Workloads via TEE

The paper also uses pretrained models with imagenet weights which can be found at - https://keras.io/api/applications/

The distillation folder contains scripts for training a model and distilling it using vanilla distillation or distillation based fine tuning.

The Generative Training folder contains scripts for training the detector and corrector using FIDES' GAN framework. 

The sgx_demo folder contains scripts for testing tflite models in sgx using graphene library

