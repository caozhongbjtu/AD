# AD

# Environment  
python 3.9.16  
torch==2.0.0  
numpy==1.26.4  
pandas==1.5.3  
scikit-learn==1.2.2  
sktime==0.33.0  

# Datasets

Datasets can be downloaded from: https://drive.google.com/file/d/1XBEDLHQV8au5SyLqVneBNkqEOBIeUgDh/view?usp=sharing 

After downloading, the data structure should be: all_datasets/anomaly_detection/*, where * represents the dataset names .
# Running Scripts  

The script_forecast.sh file contains all running commands. You can directly execute:
```bash
sh train.sh
```
Alternatively, you can run an individual script, for example: 
```bash 
python train.py --data PSM --win_size 100 --batch_size 32 --epochs 10
```
