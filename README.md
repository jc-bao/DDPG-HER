# DDPG With HER  (Pytorch Version)

This is a pytorch implementation of Deep Deterministic Policy Gradient and Hindsight Experience Replay

## Acknowledgement:

* [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
* [Hindsight Experience Replay](https://github.com/TianhongDai/hindsight-experience-replay)

## Use Cases

### Train

Ordinary Training

```python
python train.py --env-name='XarmFetch-v0' --n-epochs 100 --n-cycles 500
```

Use MPI

```python
mpirun -np 32 python -u train.py --env-name='XarmReach-v0' --n-epochs 10
mpirun -np 32 python -u train.py --env-name='XarmPDFetch-v0' 2>&1 | tee pick.log
```

### Show Demo

```python
python demo.py --env-name 'XarmPDFetch-v0'
```

## Demo

| FetchReach-v1(PyBullet)                | XarmPickAndPlace-v0(PyBullet)                                | XarmPDPickAndPlace-v0(PyBullet)                              |
| -------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![(good)reach](../pic/(good)reach.gif) | ![Large GIF (320x320)](https://tva1.sinaimg.cn/large/008i3skNgy1gsxjlnnjudg308w08wu0x.gif) | ![Large GIF (428x428)](https://tva1.sinaimg.cn/large/008i3skNgy1gsxim2p5ltg30bw0bwb2l.gif) |
| the number of epochs = 10              | the number of epochs = 50                                    | the number of epochs = 50                                    |

## Training Result

| XarmReach-v0(PyBullet)                                       | XarmPickAndPlace-v0(PyBullet)                                | XarmPickAndPlace-v0(PyBullet)                                |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20210729083753727](https://tva1.sinaimg.cn/large/008i3skNgy1gsxiuqp1nvj30d907smx4.jpg) | ![image-20210729083658506](https://tva1.sinaimg.cn/large/008i3skNgy1gsxitsp03lj30db07umxc.jpg) | ![image-20210729084952181](https://tva1.sinaimg.cn/large/008i3skNgy1gsxj77ii08j30ep092jrn.jpg) |

