# Deep Reinforcement Learning Practice with Google Colab

## Day 1: Value-Based RL
- 실습 교육 내용:
    - DQN
- 실습 내용 (못하면 숙제):
    1. Prioritized Experience Replay (DQN+) [[Schaul et al. ICLR 2016]](https://arxiv.org/pdf/1511.05952.pdf)
    2. Dueling
network architectures for deep reinforcement learning (DQN++) [[Wang et al. ICML 2016]](http://proceedings.mlr.press/v48/wangf16.pdf)

## Day 2: Policy-Based RL
- 실습 교육 내용:
    - Trust region policy optimization (TRPO) [[Schulman et al. ICML 2015]](http://proceedings.mlr.press/v37/sch제ulman15.pdf)
- 실습 내용 (못하면 숙제):
    - Proximal Policy Optimization (PPO) [[Schulman et al. Arxiv 2017]](https://arxiv.org/pdf/1707.06347.pdf)
     
## Day 3: Transfer Learning
- 실습 교육 내용:
    - Model-Agnostic Meta-Learning (MAML) [[Finn et al. ICML 2017]](https://arxiv.org/pdf/1703.03400.pdf)
- 실습 내용 (못하면 숙제):
    -  Tensorflow로 구현된 MAML을 Theano로 구현.


## Mujoco Installation
```bash
# Files required for your installation should be located at .../rllab/vendor/mujoco
# File names: libglfw.so.3, libmujoco131.so, mjkey.txt, mjpro131_linux.zip
# Except the lisence key (mjkey.txt), it is all included in our repository
# Place your lisence key in .../rllab/vendor/mujoco
# And follow the installation guide below

$ cd .../rllab/scripts
$ rllab/scripts $ ./setup_mujoco.sh 
# Enter the path to mjpro131_linux.zip which is: ../vendor/mujoco/mjpro131_linux.zip
# Enter the path to the lisence key which is: ../vendor/mujoco/mjkey.txt

# Test if Mujoco is installed in a correct way:
$ cd .../rllab/examples 
.../rllab/examples $ cp trpo_swimmer.py ../trpo_swimmer.py
.../rllab/examples $ cd ..
.../rllab $

# change the default of one of the inputs of the rollout() method named "animated" to True 
# open .../rllab/rllab/sampler/utils.py
# change the default value of the input here and save

# Run the example code that trains TRPO agent in the Swimmer environment of Mujoco
# You should see the animation of the agent moving around (Its early training stage, so it wouldn't move much)
.../rllab $ python trpo_swimmer.py  


```



## Preparation Progress
0. ~~Colab에서 anaconda env로 rllab 실행하는걸 마무리~~. ~~Anaconda 안 쓰고, CUDNN을 따로 깔아서 GPU로 rllab 코드를 돌릴 수 있 colab ipython notebook도 따로 만듬~~.
1. ~~learned parameter, average return 저장 옵션 어떻게 주는지 확인~~ (trpo_cartpole.py, trpo_swimmer.py)
2. ~~iteration  별로 average return plot (pyplot)  (trpo_cartpole.py, trpo_swimmer.py)~~
3. code 블록 의미 있게 나눠서 따로 실행 가능하게 colab에 써보기
4. distral, maml 분석. rllab 사용 가능한지 보기.. (내 기억에는 maml rllab으로 구현됐던거 같음)
5. ~~env observation normalization function 수정 (거의 모든 알고리즘에 적용되기 때문에 수정해야 함)~~ (교수님 rllab파일 사용)
(baselines, rllab trpo가 같은 성능을 보이게)
6. ~~gym environment에 video recording=True옵션 줘서 님mp4 저장하는 것 시도해볼 것~~
7. DQN+, DQN++, MAML 코드 돌려보고 colab에 추가
    - DQN, DQN+, DQN++  비교 결과: https://arxiv.org/pdf/1710.02298.pdf
    - DQN+, DQN++ 언급한 논문: https://arxiv.org/pdf/1604.00289.pdf

        - 현재 최종 결과물. TRPO GPU 돌리고, env rendering, average return plot 까지 했음. (별도의 cudnn 설치 없이 잘 돌아감.): https://colab.research.google.com/drive/1Pfhane7wLW7jr2jyqhuGk65ZShuYNEyD#scrollTo=_KwYCkcKiV2N
    
    
    
    