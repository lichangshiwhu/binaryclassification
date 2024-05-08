#!/bin/bash  

#run expand
python main.py -F ./script240520/test_breastcancer_LogisticLoss.yml -O breastcancer_LogisticLoss &
python main.py -F ./script240520/test_breastcancer_sigmoidloss.yml -O breastcancer_sigmoidloss &
python main.py -F ./script240520/test_earlyStageDiabetesRiskPrediction_LogisticLoss.yml -O earlyStageDiabetesRiskPrediction_LogisticLoss &
python main.py -F ./script240520/test_earlyStageDiabetesRiskPrediction_SigmoidLoss.yml -O earlyStageDiabetesRiskPrediction_SigmoidLoss &
python main.py -F ./script240520/test_housevotes_LogisticLoss.yml -O housevotes_LogisticLoss &
python main.py -F ./script240520/test_housevotes_SigmoidLoss.yml -O housevotes_SigmoidLoss &
python main.py -F ./script240520/test_ionosphere_LogisticLoss.yml -O ionosphere_LogisticLoss &
python main.py -F ./script240520/test_ionosphere_SigmoidLoss.yml -O ionosphere_SigmoidLoss &
python main.py -F ./script240520/test_musk_LogisticLoss.yml -O musk_LogisticLoss &
python main.py -F ./script240520/test_musk_SigmoidLoss.yml -O musk_SigmoidLoss &
python main.py -F ./script240520/test_statlog_LogisticLoss.yml -O statlog_LogisticLoss &
python main.py -F ./script240520/test_statlog_SigmoidLoss.yml -O statlog_SigmoidLoss &

#run depth
python main.py -F ./script240520/test_breastcancer_LogisticLoss_depth.yml -O breastcancer_LogisticLoss_depth &
python main.py -F ./script240520/test_breastcancer_sigmoidloss_depth.yml -O breastcancer_sigmoidloss_depth &
python main.py -F ./script240520/test_earlyStageDiabetesRiskPrediction_LogisticLoss_depth.yml -O earlyStageDiabetesRiskPrediction_LogisticLoss_depth &
python main.py -F ./script240520/test_earlyStageDiabetesRiskPrediction_SigmoidLoss_depth.yml -O earlyStageDiabetesRiskPrediction_SigmoidLoss_depth &
python main.py -F ./script240520/test_housevotes_LogisticLoss_depth.yml -O housevotes_LogisticLoss_depth &
python main.py -F ./script240520/test_housevotes_SigmoidLoss_depth.yml -O housevotes_SigmoidLoss_depth &
python main.py -F ./script240520/test_ionosphere_LogisticLoss_depth.yml -O ionosphere_LogisticLoss_depth &
python main.py -F ./script240520/test_ionosphere_SigmoidLoss_depth.yml -O ionosphere_SigmoidLoss_depth &
python main.py -F ./script240520/test_musk_LogisticLoss_depth.yml -O musk_LogisticLoss_depth &
python main.py -F ./script240520/test_musk_SigmoidLoss_depth.yml -O musk_SigmoidLoss_depth &
python main.py -F ./script240520/test_statlog_LogisticLoss_depth.yml -O statlog_LogisticLoss_depth &
python main.py -F ./script240520/test_statlog_SigmoidLoss_depth.yml -O statlog_SigmoidLoss_depth &

#run width
python main.py -F ./script240520/test_breastcancer_LogisticLoss_width.yml -O breastcancer_LogisticLoss_width &
python main.py -F ./script240520/test_breastcancer_sigmoidloss_width.yml -O breastcancer_sigmoidloss_width &
python main.py -F ./script240520/test_earlyStageDiabetesRiskPrediction_LogisticLoss_width.yml -O earlyStageDiabetesRiskPrediction_LogisticLoss_width &
python main.py -F ./script240520/test_earlyStageDiabetesRiskPrediction_SigmoidLoss_width.yml -O earlyStageDiabetesRiskPrediction_SigmoidLoss_width &
python main.py -F ./script240520/test_housevotes_LogisticLoss_width.yml -O housevotes_LogisticLoss_width &
python main.py -F ./script240520/test_housevotes_SigmoidLoss_width.yml -O housevotes_SigmoidLoss_width &
python main.py -F ./script240520/test_ionosphere_LogisticLoss_width.yml -O ionosphere_LogisticLoss_width &
python main.py -F ./script240520/test_ionosphere_SigmoidLoss_width.yml -O ionosphere_SigmoidLoss_width &
python main.py -F ./script240520/test_musk_LogisticLoss_width.yml -O musk_LogisticLoss_width &
python main.py -F ./script240520/test_musk_SigmoidLoss_width.yml -O musk_SigmoidLoss_width &
python main.py -F ./script240520/test_statlog_LogisticLoss_width.yml -O statlog_LogisticLoss_width &
python main.py -F ./script240520/test_statlog_SigmoidLoss_width.yml -O statlog_SigmoidLoss_width &

#run in pratical 
python main.py -F ./script240520/test_catanddog.yml -O test_catanddog &

wait

echo "Finsh all"

# copy to *__depth.yml
cp test_breastcancer_LogisticLoss.yml test_breastcancer_LogisticLoss_depth.yml
cp test_breastcancer_sigmoidloss.yml test_breastcancer_sigmoidloss_depth.yml
cp test_housevotes_LogisticLoss.yml test_housevotes_LogisticLoss_depth.yml
cp test_housevotes_SigmoidLoss.yml test_housevotes_SigmoidLoss_depth.yml
cp test_musk_LogisticLoss.yml test_musk_LogisticLoss_depth.yml
cp test_musk_SigmoidLoss.yml test_musk_SigmoidLoss_depth.yml
cp test_ionosphere_LogisticLoss.yml test_ionosphere_LogisticLoss_depth.yml
cp test_ionosphere_SigmoidLoss.yml test_ionosphere_SigmoidLoss_depth.yml
cp test_earlyStageDiabetesRiskPrediction_LogisticLoss.yml test_earlyStageDiabetesRiskPrediction_LogisticLoss_depth.yml
cp test_earlyStageDiabetesRiskPrediction_SigmoidLoss.yml test_earlyStageDiabetesRiskPrediction_SigmoidLoss_depth.yml
cp test_statlog_LogisticLoss.yml test_statlog_LogisticLoss_depth.yml
cp test_statlog_SigmoidLoss.yml test_statlog_SigmoidLoss_depth.yml
