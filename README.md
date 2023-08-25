# CP-gait-prediction

This software predicts gait trajectories (hip, knee, and ankle angles of the left and right leg) for typically developing children and children with Cerebral Palsy (CP). It uses deep learning networks (Fully Connected Networks, Convolutional Neural Networks, Long Short Term Memory Networks, and Transformers) utilising the `pytorch` package.

<img src="https://github.com/rkolaghassi/CP-gait-prediction/assets/46927648/2e2c5d7d-ac45-446a-9fa3-bf14ac19d129"  width="600" height="250">

## Workflow

This repository contains three procedures:

1. Hyperparameter optimisation of the deep learning networks 
2. Training of the networks with gait patterns of typically developing children 
3. Evaluating the trained networks on typically developing gait and the gait of children with Cerebral Palsy

Evaluating the networks involves predicting the stability of predictions in the short-term (one-step-ahead predictions) and in the long-term (200 recursively predicted time-steps). It also involves assessing the robustness of the networks under varying levels of Gaussian Noise (1-5%).

These procedures are implemented in the following files respectively:

1. `hyperparameter-optimisation.ipynb`
2. `Training.ipynb`
3. `Evaluation.ipynb`


## Citation 
Parts of this code have been used to develop the following research paper: "Deep Learning Models for Stable Gait Prediction Applied to Exoskeleton Reference Trajectories for Children With Cerebral Palsy", available at: https://ieeexplore.ieee.org/document/10058948

```
@article{kolaghassi2023deep,
  title={Deep Learning Models for Stable Gait Prediction Applied to Exoskeleton Reference Trajectories for Children With Cerebral Palsy},
  author={Kolaghassi, Rania and Marcelli, Gianluca and Sirlantzis, Konstantinos},
  journal={IEEE Access},
  volume={11},
  pages={31962--31976},
  year={2023},
  publisher={IEEE}
}
```

## Credits
This research was supported by the Interreg 2 Seas Programme 2014–2020, European Regional Development Fund (ERDF) under Contract2S05-038 (project links: www.motion-interreg.eu and www.interreg2seas.eu/en/MOTION). The data used to train these models has been collected and provided by Canterbury Christ Church University, and Chailey Clinical Services, both partners of the MOTION project. 

