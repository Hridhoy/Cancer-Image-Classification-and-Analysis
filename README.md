📊 Datasets

Skin Cancer: HAM10000 Dataset

10,015 dermatoscopic images

7 classes (Melanoma, Nevi, BKL, BCC, AKIEC, VASC, DF)

Breast Cancer: Wisconsin Diagnostic Breast Cancer Dataset

569 samples

30 features

Binary diagnosis: Malignant (1) vs Benign (0)

⚙️ Installation

Clone the repository and install dependencies:

git clone https://github.com/yourusername/cancer-diagnostics-ml.git
cd cancer-diagnostics-ml
pip install -r requirements.txt


Main libraries:

Python 3.8+

TensorFlow / Keras

Scikit-learn

Pandas, NumPy, Seaborn, Matplotlib

🚀 Usage
Skin Cancer CNN
cd skin_cancer
jupyter notebook cnn_skin_cancer.ipynb

Breast Cancer ML Models
cd breast_cancer
jupyter notebook breast_cancer_ml.ipynb

✅ Results
Skin Cancer (CNN)

Accuracy: ~85% (validation)

Strengths: Performs well on frequent classes

Weaknesses: Struggles with rare lesion types

Breast Cancer (ML Models)

Logistic Regression: ~94%

Decision Tree: ~92%

Random Forest: ~95%

SVM: ~96.4% (best)

🌍 Ethical & Professional Considerations

These models are for research and educational purposes only.

They should not be used as standalone diagnostic tools.

Clinical deployment requires regulatory approval, ethical validation, and large-scale clinical trials.
