## Pytorch code for GAN models
This is the pytorch implementation of 5 different GAN models using same convolutional architecture.
- GAN
- CGAN
- DCGAN (Deep convolutional GAN)
- WGAN-CP (Wasserstein GAN using weight clipping)
- WGAN-GP (Wasserstein GAN using gradient penalty)



## Dependecies
The prominent packages are:

* numpy
* scikit-learn
* tensorflow 2.5.0
* pytorch 1.8.1
* torchvision 0.9.1

To install all the dependencies quickly and easily you should use __pip__

```python
pip install -r requirements.txt
```



 *Training*
 ---
Running training of GAN model on BMG dataset:


```
python main.py --model GAN --is_train True --download False   --dataroot datasets/BMG/   --dataset alloys  --epochs 100 --cuda True --batch_size 1
```

Running training of CGAN model on BMG dataset:

```
ython main.py --model CGAN --is_train True --download False   --dataroot datasets/BMG/   --dataset alloys_c  --epochs 100 --cuda True --batch_size 1
```

Start tensorboard:

```
tensorboard --logdir ./logs/
```


