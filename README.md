# CovidEmo

This is the page for our LREC 2022 paper [Emotion analysis and detection during COVID-19](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.750.pdf). If you use this dataset, please cite our paper:

```bibtex
@inproceedings{Sosea2022EmotionAA,
  title={Emotion analysis and detection during COVID-19},
  author={Tiberiu Sosea and Chau Thi Minh Pham and Alexander Tekle and Cornelia Caragea and Junyi Jessy Li},
  booktitle={LREC},
  year={2022}
}
```
## Abstract

Understanding emotions that people express during large-scale crises helps inform policy makers and first responders about
the emotional states of the population as well as provide emotional support to those who need such support. We present
COVIDEMO, a dataset of ∼3,000 English tweets labeled with emotions and temporally distributed across 18 months. Our
analyses reveal the emotional toll caused by COVID-19, and changes of the social narrative and associated emotions over
time. Motivated by the time-sensitive nature of crises and the cost of large-scale annotation efforts, we examine how well large
pre-trained language models generalize across domains and timeline in the task of perceived emotion prediction in the context
of COVID-19. Our analyses suggest that cross-domain information transfers occur, yet there are still significant gaps. We
propose semi-supervised learning as a way to bridge this gap, obtaining significantly better performance using unlabeled data
from the target domain.

The splits used in the paper can be found in `binary_plits` directory. To reproduce the results in the paper with transfer from GoEmotions dataset, run:

```
python train.py --model <huggingface_model>
```

Please use `digitalepidemiologylab/covid-twitter-bert` for the best `CTBERT` results from the paper. To reproduce the results using HurricaneEmo, download the dataset from https://github.com/shreydesai/hurricane, then place it in the same format (HurricaneEmo is not directly downloadable through HuggingFace). The training script will generate the results in a human-readable `json` file.

If you have any questions or issues, please create an `Issue` in this repository.
