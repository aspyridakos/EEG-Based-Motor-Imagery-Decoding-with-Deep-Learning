# EEG-Based Motor Imagery Decoding with Deep Learning
Last updated: April 20th, 2024

## **Introduction**

The goal of this project is to find effective and efficient machine learning models to decode EEG motor imagery data. This project if successful could positively contribute to creating new technologies to support those with disabilities such as ALS, Parkinsons, quadriplegia, etc. whose motor function may be impeded by physical or neurological limitations. Some of the main challenges faced in decoding EEG signals is that they can be impacted by other external stimuli so isolating movements from other reactions to a test subject's environment can be quite difficult. Additionally, everyone's EEG signals are a little bit different, for this reason we need machine learning algorithms to find patterns in these diverse datasets and filter out the noise in the data model which would be otherwise impossible to do manually in a reasonable amount of time.


## **Abstract**
   This research aims to find machine learning models tasked to decode EEG motor imagery data with more accuracy than the provided benchmarks of [existing EEGNet models](https://github.com/speechbrain/benchmarks/blob/main/benchmarks/MOABB/models/EEGNet.py) in SpeechBrain-MOAB's library for the [BNCI2014_001](https://neurotechx.github.io/moabb/generated/moabb.datasets.BNCI2014_001.html#moabb.datasets.BNCI2014_001) dataset. The main methods used were (1) preprocessing data into three sets; a trainng set, a validation set, and a test set, (2) defining the model, (3) finding optimal hyperparameters, (4) training the model, and (5) running the model on the entire dataset to determine new model accuracy benchmarks. In the end, the model with the best performance, though only  marginally better than the [original EEGNet benchmarks](https://github.com/speechbrain/benchmarks/tree/main/benchmarks/MOABB#-results) (+1%) was `CustomModel3`, which implemented depthwise separable 2D convolutions instead of the standard 2D convolutional layers. Upon comparing all the custom models created, it was determined that simpler architectures outperformed more complex architectures when benchmarking the different models. No conclusions could be drawn as to whether `CustomModel3` was indeed better than `EEGNet` as further experimentation is required.

## Architecture Diagrams
### EEGNet
![image](https://github.com/aspyridakos/Speechbrain-MOABB-EEG-Processing/assets/71853151/f66001f7-6190-4259-b241-cca59db437a1)


### CustomModel3
![image](https://github.com/aspyridakos/Speechbrain-MOABB-EEG-Processing/assets/71853151/a2a7ede3-052f-42c3-b36a-8bfad514790c)



## Results and Discussion

### Results Overview

| Model                        | Dataset    | Task        | Hyperparams file                                        | Training strategy   | Performance | GPU                 |
|------------------------------|------------|-------------|---------------------------------------------------------|---------------------|------------------------|---------------------|
|Release 23-10-02 (SpeechBrain)|BNCI2014_001|Motor imagery|/MotorImagery/BNCI2014001/EEGNet.yaml                    |leave-one-session-out|0.731559±0.003888 (test set --> all subjects) |1xNVIDIA V100 (16 GB)|
|Custom model 1                |BNCI2014_001|Motor imagery|/content/drive/MyDrive/EEG_Project/CustomModel1.yaml     |leave-one-session-out|5.18e-01 (validation set -->one subject) |1xNVIDIA V100 (16 GB)|
|Custom model 2                |BNCI2014_001|Motor imagery|/content/drive/MyDrive/EEG_Project/CustomModel2.yaml     |leave-one-session-out|8.75e-01 (validation set -->one subject) |1xNVIDIA V100 (16 GB)|
|Custom model 3                |BNCI2014_001|Motor imagery|/content/drive/MyDrive/EEG_Project/CustomModel3_best.yaml|leave-one-session-out|0.745945 ± 0.00236 (test set --> all subjects) |1xNVIDIA V100 (16 GB)|

When comparing the results achieved in the table above, it is important to note that while the accuracy of the validation set appears to be higher for CustomModel2, in practice, on the test set it achieved a accuracy much closer to 60% and was therefore underperforming relative to the results of CustomModel3 which had an accuracy of approximately 74% in the test set on all subjects. The results of CustomModel3 and EEGNet's benchmarks were quite similar, and even though CustomModel3 beat the benchmark by 1%, it cannot be concluded that this model is indeed and improvement overall as the results are so close that starting with a different seed for instance may result in the opposite outcome for accuracy rank. CustomModel1 also was not run on all subjects as the accuracy was well under 60% and therefore not worth the computational resources as it certainly would not outperform the benchmarks established with EEGNet.

From these experiments, considering how poorly CustomModel3 performed primarily due to its addition of another convolutional layer, we can conclude that EEG decoding for motor imagery can benefit from model simplicity. Additionally, while it cannot be confirmed with this experiment alone that depthwise separable 2D convolutional layers are better than the standard 2D convolutional layer, it at least shows promise as an alternative that does not negatively affect accuracy of models in a  big way.

Below is a table of the results breakdown for each run in CustomModel3. One key takeaway to note is the model's stability throughout the runs. There is no unpredictable behaviour, nor does the model exhibit signs of significant overtraining across runs.

#### Custom Model 3 Results       
                                                                   
| Run | Training Accuracy | Test Accuracy   | Aggregated Accuracy Results                |
|-----|-------------------|-----------------|--------------------------------------------|
| 1   | 0.7531 ± 0.0712   | 0.7531 ± 0.0712 | 0.7550154320987654 ± 0.0019290123456790487 |
| 2   | 0.7411 ± 0.0744   | 0.7407 ± 0.0731 | 0.7409336419753088 ± 0.0001929012345678882 |
| 3   | 0.7442 ± 0.0744   | 0.7369 ± 0.0767 | 0.7405478395061729 ± 0.003665123456790209  |
| 4   | 0.7539 ± 0.078    | 0.7392 ± 0.0859 | 0.7465277777777777 ± 0.007330246913580307  |
| 5   | 0.7485 ± 0.0702   | 0.7434 ± 0.0789 | 0.7459490740740742 ± 0.002507716049382658  |
| 6   | 0.7569 ± 0.0736   | 0.7477 ± 0.0738 | 0.7523148148148149 ± 0.00462962962962965   |
| 7   | 0.7481 ± 0.0827   | 0.7423 ± 0.0751 | 0.7451774691358024 ± 0.0028935185185185457 |
| 8   | 0.7504 ± 0.0804   | 0.7477 ± 0.0785 | 0.7490354938271605 ± 0.001350308641975384  |
| 9   | 0.7434 ± 0.0633   | 0.7407 ± 0.0762 | 0.7420910493827162 ± 0.0013503086419752175 |
| 10  | 0.7434 ± 0.0754   | 0.7404 ± 0.0746 | 0.7418981481481481 ± 0.0015432098765431057 |

## **References**
[1] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance, “EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces,” J. Neural Eng., vol. 15, no. 5, p. 056013, Oct. 2018, doi: 10.1088/1741-2552/aace8c.

[2] Speechbrain, “benchmarks/benchmarks/MOABB at main · speechbrain/benchmarks,” GitHub. https://github.com/speechbrain/benchmarks/tree/main/benchmarks/MOABB

[3] “moabb.datasets.BNCI2014_001 — moabb 1.0.0 documentation.” https://neurotechx.github.io/moabb/generated/moabb.datasets.BNCI2014_001.html#moabb.datasets.BNCI2014_001

```
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```
