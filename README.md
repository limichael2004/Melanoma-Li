
# Deep Learning Framework for Skin Lesion Classification

This Python code implements a  deep learning framework for skin lesion classification using nested cross-validation and ensemble methods. Here's a detailed breakdown of what the code does:

# Select Metrics: 

<img width="340" alt="Screenshot 2025-05-03 at 9 49 46 PM" src="https://github.com/user-attachments/assets/30457dfe-dd86-4728-874f-44126e00fdeb" />

# Images

<img width="1275" alt="Screenshot 2025-05-03 at 9 53 13 PM" src="https://github.com/user-attachments/assets/0910963c-cf56-4faf-8507-f0efb75aacc5" />

<img width="1278" alt="Screenshot 2025-05-03 at 9 54 11 PM" src="https://github.com/user-attachments/assets/5328081f-d3f4-4bb3-84bf-a86eaa876f83" />


# Overall Purpose
The code creates a sophisticated machine learning system to classify skin lesions in medical images (likely distinguishing between malignant and benign lesions) with high accuracy and reliability. It uses advanced techniques including:

Nested k-fold cross-validation for robust performance estimation

Bayesian hyperparameter optimization with Optuna

Ensemble learning combining multiple neural network architectures

Dynamic weighting of model predictions

Augmentation techniques including custom image mixing within classes

Comprehensive evaluation metrics and visualization

# Key Components
1. Data Handling and Augmentation

Uses a custom ClassAwareDataset that extends PyTorch's ImageFolder

Implements a novel ImageMixTransform that intelligently combines images from the same class using various strategies (vertical, horizontal, diagonal splits, etc.)

Applies standard augmentation techniques (rotation, flips, color jittering)

2. Model Architecture

Creates an ensemble of three pre-trained CNN architectures: ResNet101, EfficientNet-B4, and DenseNet121
Customizes each model with:

Optional attention modules to enhance feature importance

Configurable dropout for regularization

Adjustable hidden layers

Freezing options for transfer learning



3. Ensemble Learning

Uses a DynamicWeight module to learn optimal weights for each model's predictions

Combines predictions intelligently rather than simple averaging

4. Nested Cross-Validation

Implements nested k-fold cross-validation (10 outer folds, 9 inner folds)

Inner folds optimize hyperparameters with Optuna

Outer folds evaluate generalization performance


5. Hyperparameter Optimization

Employs Bayesian optimization (via Optuna) to find optimal hyperparameters

Optimizes data augmentation, model architecture, and training parameters

Uses TPE sampler and median pruner for efficient optimization

6. Comprehensive Metrics and Visualization

Calculates and visualizes numerous performance metrics:

Accuracy, precision, recall, F1 score, AUC-ROC, MCC

Confusion matrices and ROC curves

Probability distributions

Threshold analysis

Creates overlay plots for direct comparison of models and folds

7. Final Evaluation

Evaluates best models from each fold on an external test set

Creates and evaluates an ensemble of all fold models

Generates comprehensive reports and visualizations



# Output and Results

This is a highly sophisticated, research-grade machine learning pipeline for medical image classification that incorporates state-of-the-art techniques for robust performance evaluation and optimization.



## Acknowledgements

Thank you to my mentor Dr. Jianneng Li, who gave me the freedom to explore my interests in deep learning and computational research.

Thank you to the Notre Dame College of Science and donors for their support through the Student Initiated Grant.

Thank you to the Notre Dame Center for Research Computing (CRC) for allowing me access to their GPU cluster and computational resources. Special acknowledgement for Mr. Dodi for his help in the preliminary stages of the project!

Special thanks to my friend Kevin Xue as well for his help and advice.

Also thank you to my mom and dad!
