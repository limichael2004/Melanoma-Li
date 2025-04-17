
# Artifical Intelligence Based Detection of Skin Cancer by Michael Li

Hi, my name is Michael and I am a current junior at the University of Notre Dame majoring in Neuroscience.

I taught myself to code in May of 2024 and I recently build an Artifical Intelligence (Deep Learning) framework to detect benign or malignant skin lesions.

The novelty of this framework is an dynamic weighted ensemble model (Resnet101, EfficientNet-B7, and DenseNet121) with Bayesian Optimization over 50 trials for >15 hyperparameters. Additionally, there is a custom threshold implementation for binary classification, along with nested cross validation with 5 outer folds and inner folds.

The model was trained and nested cross-validated on the HAM10000 dataset, before undergoing a seperate external test set by Memorial Sloan Kettering.

*Please note that there is a minor bug in the main code (combo.py), which is adressed in the testcombo.py and ensemble_folds_overlaid_graphs.py


## Results
Here are some select metrics:

NESTED CROSS-VALIDATION RESULTS:

EXTERNAL TEST SET RESULTS (AVERAGE ACROSS 5 FOLDS):

Average Auc: 0.9208 ± 0.0079

Average Accuracy: 0.8824 ± 0.0056

Average Specificity: 0.9260 ± 0.0075


DYNAMIC WEIGHTED ENSEMBLE TEST RESULTS:

Optimal Threshold: 0.3386

Auc: 0.9462

Accuracy: 0.8962

Sensitivity: 0.8208

Specificity: 0.9154


## Acknowledgements

Thank you to my mentor Dr. Jianneng Li, who gave me the freedom to explore my interests in deep learning and computational research.

Thank you to the Notre Dame College of Science and donors for their support through the Student Initiated Grant.

Thank you to the Notre Dame Center for Research Computing (CRC) for allowing me access to their GPU cluster and computational resources. Special acknowledgement for Mr. Dodi for his help in the preliminary stages of the project!

Special thanks to my friend Kevin Xue as well for his help and advice.

Also thank you to my mom and dad!
