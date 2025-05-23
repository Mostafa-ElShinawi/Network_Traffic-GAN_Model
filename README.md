#     Network_Traffic-GAN_Model
# 🚦 GAN for Malicious & Benign Traffic Generation  

## 🔍 Overview  
A **Generative Adversarial Network (GAN)** trained to generate realistic network traffic (both malicious and benign) for cybersecurity research.
 ## **Key Features**  
  - Customizable GAN architecture  
  - Handles IPs, protocols, numerical features  
  - Preserves original traffic statistics  
  - Balanced malicious/benign generation  

## 📂 Files  
- **Original dataset**: `data/raw/dataset.csv` (68.7 MB)  
- **Generated samples**: `data/generated/synthetic_traffic.csv`  
- **Code**: `src/train_gan.py`  
- **Project PDF**: `docs/projectdescription.pdf`  

## 🛠️ Setup  
1. Install PyTorch:  
```bash
pip install torch torchvision
pip install pandas scikit-learn numpy
python src/train_gan.py

Data
 Input: data/raw/dataset.csv
 Output: data/generated/synthetic_traffic.csv
    
