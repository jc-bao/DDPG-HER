# DDPG With HER  (Pytorch Version)

This is a pytorch implementation of Deep Deterministic Policy Gradient and Hindsight Experience Replay

## Acknowledgement:

* [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
* [Hindsight Experience Replay](https://github.com/TianhongDai/hindsight-experience-replay)

## Use Cases

### Train

```python
python train.py --env-name='XarmFetch-v0' --n-epochs 100 --n-cycles 500
```

### Show Demo

```python
python demo.py --env-name 'XarmFetch-v0'
```

## Demo

| FetchReach-v1                                                |      | XarmPickAndPlace-v0(PyBullet)                                |
| ------------------------------------------------------------ | ---- | ------------------------------------------------------------ |
| ![reach-mujoco](https://tva1.sinaimg.cn/large/008i3skNly1gswqpam9pmg30bw0bwb2d.gif) |      | ![PickAndPlace-bullet](https://tva1.sinaimg.cn/large/008i3skNly1gswrjjh0frg30bw0bwu13.gif) |

## Training Result

| FetchReach-v1                                                |      | XarmPickAndPlace-v0(PyBullet) |
| ------------------------------------------------------------ | ---- | ----------------------------- |
| ![Success Rate (2)](../../../../../Downloads/Success%20Rate%20(2).svg) |      |                               |

## 