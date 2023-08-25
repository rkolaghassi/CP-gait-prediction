# CP-gait-prediction

This software predicts gait trajectories (hip, knee, and ankle angles of the left and right leg) for typically developing (healthy) children and children with Cerebral Palsy. It uses deep learning networks (Fully Connected Networks, Convolutional Neural Networks, Long Short Term Memory Networks, and Transformers) through `pytorch` packages.


## Workflow

This repository contains three procedures:

1. Hyperparameter optimisation of the deep learning networks 
2. Training the networks with gait patterns of typically developing children 
3. Evaluating the trained networks on typically developing gait and the gait of children with Cerebral Palsy

Evaluating the networks involves predicting the stability of predictions in the short-term (one-step-ahead predictions) and in the long-term (200 recursively predicted time-steps). It also involves assessing the robustness of the networks under varying levels of Gaussian Noise (1-5%).

These procedures are implemented in the following files respectively:

1. hyperparameter-optimisation.ipynb
2. Training.ipynb
3. Evaluation.ipynb


## Citation 
Parts of this code have been used to develop the following research paper: "Deep Learning Models for Stable Gait Prediction Applied to Exoskeleton Reference Trajectories for Children With Cerebral Palsy", available at: https://www.researchgate.net/publication/369051668_Deep_Learning_Models_for_Stable_Gait_Prediction_Applied_to_Exoskeleton_Reference_Trajectories_for_Children_with_Cerebral_Palsy 

## Credits
This research was supported by the Interreg 2 Seas Programme 2014–2020, European Regional Development Fund (ERDF) under Contract2S05-038 (project links: www.motion-interreg.eu and www.interreg2seas.eu/en/MOTION). The data used to train these models has been collected and provided by Canterbury Christ Church University, and Chailey Clinical Services, both partners of the MOTION project. 

