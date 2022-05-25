# Transformer-for-Nirs
a novel calibration model named SpectraTr, based on the transformer structure is proposed and used for the qualitative analysis of drug spectrum. 

# Please reference this work if any part of it is used elsewhere: DOI: 10.1142/S1793545822500213

Fu P, Wen Y, Zhang Y, et al. SpectraTr: A novel deep learning model for qualitative analysis of drug spectroscopy based on transformer structure[J]. Journal of Innovative Optical Health Sciences, 2022, 15(03): 2250021.[paper](https://www.worldscientific.com/doi/10.1142/S1793545822500213)

```
@article{fu2022spectratr,
  title={SpectraTr: A novel deep learning model for qualitative analysis of drug spectroscopy based on transformer structure},
  author={Fu, Pengyou and Wen, Yue and Zhang, Yuke and Li, Lingqiao and Feng, Yanchun and Yin, Lihui and Yang, Huihua},
  journal={Journal of Innovative Optical Health Sciences},
  volume={15},
  number={03},
  pages={2250021},
  year={2022},
  publisher={World Scientific}
}
```


# ABSTRACT:
The drug supervision methods based on near-infrared spectroscopy analysis are heavily dependent on the chemometrics model which characterizes the relationship between spectral data and drug categories.The preliminary application of convolution neural network in spectral analysis demonstrates excellent end-to-end prediction ability, but it is sensitive to the hyper-parameters of the network. The transformer is a deep learning model based on self_attention mechanism that comparable CNN in predictive performance and has an easy-to-design model structure. Hence, a novel calibration model named SpectraTr, based on the transformer structure is proposed and used for the qualitative analysis of drug spectrum.The experimental results of 7 classes of drug and 18 classes of drug show that the proposed SpectraTr model can automatically extract features from a huge number of spectra, is not dependent on pre-processing algorithms, and is insensitive to model hyperparameters.When the ratio of the training set to test set is 8:2, the prediction accuracy of the SpectraTr model reaches 100% and 99.52% respectively, which outperforms PLS_DA, SVM, SAE, and CNN.The model is also tested on a public drug data set, and achieve classification accuracy of 96.97% without pre-processing algorithm, which is 34.85%, 28.28%, 5.05% and 2.73% higher than PLS_DA, SVM, AE, and CNN respectively. The research shows that the SpectraTr model performs exceptionally well in spectral analysis and is expected to be a novel deep calibration model after AE and CNN.

![image](https://user-images.githubusercontent.com/56440282/170202683-8bd8c5fe-f1f6-460b-ae27-0b48d13bd541.png)

# How to use it:
```
step0: Install the corresponding python environment
step1: run TableVitRun.py
```

# info: We provide the test model trained in the experiments of the paper
```
you can install from 链接：https://pan.baidu.com/s/16-8xUamOZQkMb3sMLAHz1g  提取码：5jfk
then put it Transformer\model\Table and run TableVitRun.py
```
# the reslut 
<img width="2000" alt="4893faa64a630a5a46d24537c278efc" src="https://user-images.githubusercontent.com/56440282/170207738-432f7b0c-53cf-4569-8d90-afc1cb2f428b.png">
<img width="787" alt="a3c8569adcb7348b032ac2f75e97ceb" src="https://user-images.githubusercontent.com/56440282/170207771-c9c2a0eb-05d4-4869-bb33-578433ec8cd5.png">
![image](https://user-images.githubusercontent.com/56440282/170207932-884de0d1-51f0-4c32-a2fb-50dff7688644.png)
