# CP-gait-prediction

This software is designed to predict gait trajectories, specifically the hip, knee, and ankle angles of both left and right legs, for both typically developing children and children with Cerebral Palsy (CP). The prediction is accomplished through the utilisation of deep learning networks (Fully Connected Networks, Convolutional Neural Networks, Long Short Term Memory Networks, and Transformers), all of which are implemented using the `PyTorch` package.

<img src="https://github.com/rkolaghassi/CP-gait-prediction/assets/46927648/2e2c5d7d-ac45-446a-9fa3-bf14ac19d129"  width="600" height="250">

## Workflow

This repository contains three main procedures:

1. **Hyperparameter Optimisation**: This phase involves fine-tuning the parameters of the deep learning networks, a task facilitated by the `Optuna` library.
2. **Network Training**: The networks are trained using gait patterns from typically developing children.
3. **Network Evaluation**: The trained networks are evaluated using both typically developing gait patterns and the gait patterns of children with Cerebral Palsy. Evaluation encompasses analyzing network stability during short-term (one-step-ahead) predictions and long-term (200-time-step) recursive predictions. Additionally, the networks' robustness is assessed under varying levels of Gaussian Noise (ranging from 1% to 5%).

These procedures are implemented in the following files respectively:

1. `hyperparameter-optimisation.ipynb`
2. `Training.ipynb`
3. `Evaluation.ipynb`


## Citation 
Parts of this code have contributed to the development of the research paper titled: "Deep Learning Models for Stable Gait Prediction Applied to Exoskeleton Reference Trajectories for Children With Cerebral Palsy", available at: https://ieeexplore.ieee.org/document/10058948

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
This research was supported by the Interreg 2 Seas Programme 2014–2020, European Regional Development Fund (ERDF) under Contract 2S05-038 (project links: www.motion-interreg.eu and www.interreg2seas.eu/en/MOTION). The data used to train these models has been collected and provided by Canterbury Christ Church University, and Chailey Clinical Services, both partners of the MOTION project. 

