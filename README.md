# "SuperCon: Supervised Contrastive Learning for Imbalanced Skin Lesion Classification"

- Download the ISIC2020 and ISIC2019 image data from:

    https://www.isic-archive.com/#!/topWithHeader/wideContentTop/main   



- Download the train/test list:
    1. Train19+20.txt: ISIC2020 + Melanoma from ISIC2019.
    2. ISIC2020_train.txt: Only ISIC2020 training data.
    3. ISIC2020_test.csv: The mutual testing data in all experiment.
    Change the path to image files and txt/csv files.



- To reproduce the paper:
    > 1. Representation Training: use the command below:
    
        python Representation_training.py --gpu 0 --model_name resnet50 --train_model 2020only --train_batchsz 128 

    Note: requires 23GB GPU memory with train_batchsz 128. make is smaller based on your own device. But a performance degradation is expected.
    

    > 2. Classifier Fine-tuning: use the command below:
      
        python Classifier_FineTune.py --gpu 0 --epochs_finetune 5 --model_name resnet50 --train_mode 2020only --train_batchsz 128

    Note: Unlike 1. Representation Training, much lower memory required with batch size 128, but tune it based on your own situation if needed. 
    
    Before classifier fine-tuning, change the function to "TrainData" in get_data.py line 81 and line 87. Then
