# FM4MiningSegmentation  

**Benchmarking Foundation Models for Semantic Segmentation of Mining Activities in Satellite Imagery**  

## Quick Start  

Fine-tune a ResNet-50 model pretrained on ImageNet using our SmallMinesDS dataset:  

   ```bash  
   git clone https://github.com/yourusername/FM4MiningSegmentation.git  
   cd fm4_mining_segmentation  
   conda create -n fm4mining python=3.11  
   conda activate fm4mining  
   pip install -r requirements.txt   
   python ft-resnet50.py  