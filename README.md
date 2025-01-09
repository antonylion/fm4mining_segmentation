# FM4MiningSegmentation  

**Benchmarking Foundation Models for Semantic Segmentation of Mining Activities in Satellite Imagery**  

## Installation  

Follow these steps to set up the environment:  

1. **Clone the Repository**  
   ```bash  
   git clone https://github.com/yourusername/FM4MiningSegmentation.git  
   cd fm4_mining_segmentation  
   ```  

2. **Create a Conda Environment**  
   ```bash  
   conda create -n fm4mining python=3.11  
   conda activate fm4mining  
   ```  

3. **Install Dependencies**  
   ```bash  
   pip install -r requirements.txt  
   ```  

## Quick Start  

Fine-tune a ResNet-50 model pretrained on ImageNet using your dataset:  

```bash  
python ft-resnet50.py  