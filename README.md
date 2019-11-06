# deep-learnging-3d-object-dectation




### BaselineUNetModel-eva.ipynb
This notebook load the pre-trained BaselineUNetModel, and make predictions based on validation dataset, which is split by train dataset with 70/30 ratio. Then, generate a csv file called ```baseline_val_pred.csv``` which fits the submission format of the competition.

### evaluation_score.ipynb
This notebook compares the predict output and groud truth table and calculates the average score which is defined in the evaluation metrics in the report. Make sure that ```baseline_val_pred.csv``` and ```val_gt.csv``` exist and the paths to these two csv file are configured correctly.

