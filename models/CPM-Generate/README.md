---
language: 
- zh
tags:
- cpm
license: mit
datasets:
- 100GB Chinese corpus
---
# CPM-Generate

## Model description

CPM (Chinese Pre-trained Language Model) is a Transformer-based autoregressive language model, with 2.6 billion parameters and 100GB Chinese training data. To the best of our knowledge, CPM is the largest Chinese pre-trained language model, which could facilitate downstream Chinese NLP tasks, such as conversation, essay generation, cloze test, and language understanding. [[Project](https://cpm.baai.ac.cn)] [[Model](https://cpm.baai.ac.cn/download.html)] [[Paper](https://arxiv.org/abs/2012.00413)]

## Intended uses & limitations

#### How to use

```python
from transformers import TextGenerationPipeline, AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("TsinghuaAI/CPM-Generate")
model = AutoModelWithLMHead.from_pretrained("TsinghuaAI/CPM-Generate")

text_generator = TextGenerationPipeline(model, tokenizer)
text_generator('清华大学', max_length=50, do_sample=True, top_p=0.9)
```

#### Limitations and bias

The text generated by CPM is automatically generated by a neural network model trained on a large number of texts, which does not represent the authors' or their institutes' official attitudes and preferences. The text generated by CPM is only used for technical and scientific purposes. If it infringes on your rights and interests or violates social morality, please do not propagate it, but contact the authors and the authors will deal with it promptly.

## Training data

We collect different kinds of texts in our pre-training, including encyclopedia, news, novels, and Q\&A. The details of our training data are shown as follows.

| Data Source | Encyclopedia | Webpage | Story | News  | Dialog |
| ----------- | ------------ | ------- | ----- | ----- | ------ |
| **Size**    | ~40GB        | ~39GB   | ~10GB | ~10GB | ~1GB   |

## Training procedure

Based on the hyper-parameter searching on the learning rate and batch size, we set the learning rate as \\(1.5\times10^{-4}\\) and the batch size as \\(3,072\\), which makes the model training more stable. In the first version, we still adopt the dense attention and the max sequence length is \\(1,024\\). We will implement sparse attention in the future. We pre-train our model for \\(20,000\\) steps, and the first \\(5,000\\) steps are for warm-up. The optimizer is Adam. It takes two weeks to train our largest model using \\(64\\) NVIDIA V100.

## Eval results

|            | n_param | n_layers | d_model | n_heads | d_head |
|------------|-------------------:|--------------------:|-------------------:|-------------------:|------------------:|
| CPM-Small  |               109M |                  12 |                768 |                 12 |                64 |
| CPM-Medium |               334M |                  24 |              1,024 |                 16 |                64 |
| CPM-Large  |               2.6B |                  32 |              2,560 |                 32 |                80 |

We evaluate CPM with different numbers of parameters (the details are shown above) on various Chinese NLP tasks in the few-shot (even zero-shot) settings. With the increase of parameters, CPM performs better on most datasets, indicating that larger models are more proficient at language generation and language understanding. We provide results of text classification, chinese idiom cloze test, and short text conversation generation as follows. Please refer to our [paper](https://arxiv.org/abs/2012.00413) for more detailed results.


### Zero-shot performance on text classification tasks

|            |     TNEWS      |    IFLYTEK     |     OCNLI      |
| ---------- | :------------: | :------------: | :------------: |
| CPM-Small  |     0.626      |     0.584      |     0.378      |
| CPM-Medium |     0.618      |     0.635      |     0.379      |
| CPM-Large  | **0.703** | **0.708** | **0.442** |

### Performance on Chinese Idiom Cloze (ChID) dataset
|            |   Supervised   |  Unsupervised  |
|------------|:--------------:|:--------------:|
| CPM-Small  |      0.657     |      0.433     |
| CPM-Medium |      0.695     |      0.524     |
| CPM-Large  | **0.804** | **0.685** |

### Performance on Short Text Conversation Generation (STC) dataset
|                                  |     Average    |     Extrema    |     Greedy     |              Dist-1             |              Dist-2              |
|----------------------------------|:--------------:|:--------------:|:--------------:|:-------------------------------:|:--------------------------------:|
| *Few-shot (Unsupervised)* |                |                |                |                                 |                                  |
| CDial-GPT                        |      0.899     |      0.797     |      0.810     |      1,963 / **0.011**     |          20,814 / 0.126          |
| CPM-Large                        | **0.928** | **0.805** | **0.815** |      **3,229** / 0.007     | **68,008** / **0.154** |
| *Supervised*              |                |                |                |                                 |                                  |
| CDial-GPT                        |      0.933     | **0.814** | **0.826** |          2,468 / 0.008          |          35,634 / 0.127          |
| CPM-Large                        | **0.934** |      0.810     |      0.819     | **3,352** / **0.011** | **67,310** / **0.233** |




### BibTeX entry and citation info

```bibtex
@article{cpm-v1,
  title={CPM: A Large-scale Generative Chinese Pre-trained Language Model},
  author={Zhang, Zhengyan and Han, Xu, and Zhou, Hao, and Ke, Pei, and Gu, Yuxian and Ye, Deming and Qin, Yujia and Su, Yusheng and Ji, Haozhe and Guan, Jian and Qi, Fanchao and Wang, Xiaozhi and Zheng, Yanan and Zeng, Guoyang and Cao, Huanqi and Chen, Shengqi and Li, Daixuan and Sun, Zhenbo and Liu, Zhiyuan and Huang, Minlie and Han, Wentao and Tang, Jie and Li, Juanzi and Sun, Maosong},
  year={2020}
}
```# refgame_project
