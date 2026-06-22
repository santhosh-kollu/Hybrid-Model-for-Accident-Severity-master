# Explainable Hybrid Model for Accident Severity

This repository contains an implementation of an explainable hybrid model for accident severity prediction. The model combines multiple machine learning algorithms in an ensemble approach to improve prediction accuracy and provide interpretable results.

## Model Architecture

The hybrid model architecture consists of:
- LightGBM model for gradient boosting
- SVM (Support Vector Machine) model for classification
- Random Forest meta-model for ensemble learning

## Files

- `preprocess.py`: Data preprocessing script
- `train.py` and `train2.py`: Model training scripts
- `test.py` and `test_unseen.py`: Model evaluation scripts
- Model files:
  - `lgb_model.txt`: Trained LightGBM model
  - `svm_model.pkl`: Trained SVM model
  - `rf_meta_model.pkl`: Trained Random Forest meta-model
- Visualization files:
  - `model_pipeline.png`: Visual representation of the model architecture
  - Various performance comparison plots
  - Confusion matrices

## Data

The model is trained on a balanced accident dataset with features related to accident conditions, vehicle information, and environmental factors.

## Performance

The hybrid model shows improved performance compared to individual models in terms of accuracy and F1 score, particularly for minority classes in the accident severity prediction task.

---

## 👨‍💻 Developer & Contact

Developed with passion and a commitment to building robust, explainable machine learning models. 

<div align="left">
  <strong>Santhosh Kollu</strong><br>
  Machine Learning Developer & Data Scientist<br><br>
  <a href="mailto:kollusanthosh01@gmail.com">
    <img src="https://img.shields.io/badge/Email-kollusanthosh01@gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email" />
  </a>
  <a href="https://www.linkedin.com/in/santhosh-kollu-b179812b3/">
    <img src="https://img.shields.io/badge/LinkedIn-Santhosh_Kollu-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" />
  </a>
</div>