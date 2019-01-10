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


## Mujoco Local Installation
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
5. ~~env observation normalization function 수정 (거의 모든 알고리즘에 적용되기 때문에 수정해야 함)~~ (교수님 rllab파일 사용. Rllab의 Tensorflow 버전이 들어있는 sandbox에서 Theano rllab과 같은 런타임 에러가 없는지 확인해야 함)
(baselines, rllab trpo가 같은 성능을 보이게)
6. ~~gym environment에 video recording=True옵션 줘서 님mp4 저장하는 것 시도해볼 것~~
7. DQN+, DQN++, MAML 코드 돌려보고 colab에 추가
    - DQN, DQN+, DQN++  비교 결과: https://arxiv.org/pdf/1710.02298.pdf
    - DQN+, DQN++ 언급한 논문: https://arxiv.org/pdf/1604.00289.pdf

        - 현재 최종 결과물. TRPO GPU 돌리고, env rendering, average return plot 까지 했음. (별도의 cudnn 설치 없이 잘 돌아감.): https://colab.research.google.com/drive/1Pfhane7wLW7jr2jyqhuGk65ZShuYNEyD#scrollTo=_KwYCkcKiV2N
    
    
### Jonathan Ho GAIL code https://github.com/openai/imitation
- https://github.com/openai/imitation
- core: script/imitate_mj
- imitate/expert_policies/modern: contains mujoco expert policy weights
- imitate/expert_policies/classical: contains openai gym classical domain expert policy weights
- imitation/scripts/im_pipeline.py: 
    phases = {
        '0_sampletrajs': phase0_sampletrajs,
        '1_train': phase1_train,
        '2_eval': phase2_eval,
    }
     0 : sample trajs
     1 : train GAIL (imitate_mj.py 실행. seokin shell file 있음)
     2 : test GAIL
     
     
- imitate_mj 실행 스크립트
```bash
python scripts/imitate_mj.py --mode ga \
  --reward_type nn \
  --env Hopper-v1 \
  --data imitation_runs/modern_stochastic/trajs/trajs_hopper.h5 \
  --limit_trajs 25 \
  --data_subsamp_freq $1 \
  --favor_zero_expert_reward 0 \ 
  --min_total_sa 50000 \
  --sim_batch_size 1 \ 
  --max_iter 501 \
  --reward_include_time 0 \ 
  --reward_ent_reg_weight 0.0 \
  --log outputs_gail/hopper-subsampfreq$1-rewardentreg0.h5 
```

```bash
# 실행방법
$ bash [실행스크립트 이름] [arg1] [arg2] ...
```

- Test code: /scripts/vis_mj.py
```PYTHON
 parser.add_argument('policy', type=str)  <-- policy weight address
```

- where rendering function could be applied
```python
while not sim.done:
    a = policy.sample_actions(sim.obs[None,:], bool(args.deterministic))[0][0,:]
    r = sim.step(a)
    totalr += r
    l += 1
```

- With the learned policy, make 50 traj to evaluate
```python          
print 'Average return:', trajbatch.r.padded(fill=0.).sum(axis=1).mean()
```
- Select between 1)behvioral clonning & 2)GAIL
        - When GAIL is selected:
                1. reward is set with the imitation.TransitionClassifier (GAN discriminator NN)
                2. value function used in TRPO is specified (vf)... probably optimized with GAE lambda but not sure
                3. policy optimization method specified (opt): TRPO
                
```bash
    if args.mode == 'bclone':
        # For behavioral cloning, only print output when evaluating
        args.print_freq = args.bclone_eval_freq
        args.save_freq = args.bclone_eval_freq

        reward, vf = None, None
        opt = imitation.BehavioralCloningOptimizer(
            mdp, policy,
            lr=args.bclone_lr,
            batch_size=args.bclone_batch_size,
            obsfeat_fn=lambda o:o,
            ex_obs=exobs_Bstacked_Do, ex_a=exa_Bstacked_Da,
            eval_sim_cfg=policyopt.SimConfig(
                min_num_trajs=args.bclone_eval_ntrajs, min_total_sa=-1,
                batch_size=args.sim_batch_size, max_traj_len=max_traj_len),
            eval_freq=args.bclone_eval_freq,
            train_frac=args.bclone_train_frac)

    elif args.mode == 'ga':
        if args.reward_type == 'nn':
            reward = imitation.TransitionClassifier(
                hidden_spec=args.policy_hidden_spec,
                obsfeat_space=mdp.obs_space,
                action_space=mdp.action_space,
                max_kl=args.reward_max_kl,
                adam_lr=args.reward_lr,
                adam_steps=args.reward_steps,
                ent_reg_weight=args.reward_ent_reg_weight,
                enable_inputnorm=True,
                include_time=bool(args.reward_include_time),
                time_scale=1./mdp.env_spec.timestep_limit,
                favor_zero_expert_reward=bool(args.favor_zero_expert_reward),
                varscope_name='TransitionClassifier')
        elif args.reward_type in ['l2ball', 'simplex']:
            reward = imitation.LinearReward(
                obsfeat_space=mdp.obs_space,
                action_space=mdp.action_space,
                mode=args.reward_type,
                enable_inputnorm=True,
                favor_zero_expert_reward=bool(args.favor_zero_expert_reward),
                include_time=bool(args.reward_include_time),
                time_scale=1./mdp.env_spec.timestep_limit,
                exobs_Bex_Do=exobs_Bstacked_Do,
                exa_Bex_Da=exa_Bstacked_Da,
                ext_Bex=ext_Bstacked)
        else:
            raise NotImplementedError(args.reward_type)

        vf = None if bool(args.no_vf) else rl.ValueFunc(
            hidden_spec=args.policy_hidden_spec,
            obsfeat_space=mdp.obs_space,
            enable_obsnorm=args.obsnorm_mode != 'none',
            enable_vnorm=True,
            max_kl=args.vf_max_kl,
            damping=args.vf_cg_damping,
            time_scale=1./mdp.env_spec.timestep_limit,
            varscope_name='ValueFunc')

        opt = imitation.ImitationOptimizer(
            mdp=mdp,
            discount=args.discount,
            lam=args.lam,
            policy=policy,
            sim_cfg=policyopt.SimConfig(
                min_num_trajs=-1, min_total_sa=args.min_total_sa,
                batch_size=args.sim_batch_size, max_traj_len=max_traj_len),
            step_func=rl.TRPO(max_kl=args.policy_max_kl, damping=args.policy_cg_damping),
            reward_func=reward,
            value_func=vf,
            policy_obsfeat_fn=lambda obs: obs,
            reward_obsfeat_fn=lambda obs: obs,
            policy_ent_reg=args.policy_ent_reg,
            ex_obs=exobs_Bstacked_Do,
            ex_a=exa_Bstacked_Da,
            ex_t=ext_Bstacked)
```

- GAIL code running anaconda env yml file contents
```txt
name: gmmil
channels:
- conda-forge
- defaults
dependencies:
- ffmpeg=4.0=hc8c182b_0
- freetype=2.8.1=0
- gnutls=3.5.17=0
- libiconv=1.15=h470a237_1
- libidn11=1.33=0
- nettle=3.3=0
- x264=20131218=0
- backports=1.0=py27h63c9359_1
- backports.functools_lru_cache=1.5=py27_1
- backports.shutil_get_terminal_size=1.0.0=py27h5bc021e_2
- backports_abc=0.5=py27h7b3c97b_0
- bleach=2.1.3=py27_0
- bzip2=1.0.6=h9a117a8_4
- ca-certificates=2017.08.26=h1d4fec5_0
- certifi=2018.1.18=py27_0
- configparser=3.5.0=py27h5117587_0
- cycler=0.10.0=py27hc7354d3_0
- dbus=1.12.2=hc3f9b76_1
- decorator=4.3.0=py27_0
- entrypoints=0.2.3=py27h502b47d_2
- enum34=1.1.6=py27h99a27e9_1
- expat=2.2.5=he0dffb1_0
- fontconfig=2.12.6=h49f89f6_0
- functools32=3.2.3.2=py27h4ead58f_1
- futures=3.2.0=py27h7b459c0_0
- glib=2.53.6=h5d9569c_2
- gmp=6.1.2=h6c8ec71_1
- gst-plugins-base=1.12.4=h33fb286_0
- gstreamer=1.12.4=hb53b477_0
- hdf5=1.10.1=h9caa474_1
- html5lib=1.0.1=py27h5233db4_0
- icu=58.2=h9c2bf20_1
- intel-openmp=2018.0.0=hc7b2577_8
- ipykernel=4.8.2=py27_0
- ipython=5.6.0=py27_0
- ipython_genutils=0.2.0=py27h89fb69b_0
- ipywidgets=7.2.1=py27_0
- jinja2=2.10=py27h4114e70_0
- jpeg=9b=h024ee3a_2
- jsonschema=2.6.0=py27h7ed5aa4_0
- jupyter=1.0.0=py27_4
- jupyter_client=5.2.3=py27_0
- jupyter_console=5.2.0=py27hc6bee7e_1
- jupyter_core=4.4.0=py27h345911c_0
- kiwisolver=1.0.1=py27hc15e7b5_0
- libedit=3.1=heed3624_0
- libffi=3.2.1=hd88cf55_4
- libgcc-ng=7.2.0=hdf63c60_3
- libgfortran-ng=7.2.0=hdf63c60_3
- libpng=1.6.34=hb9fc6fc_0
- libsodium=1.0.16=h1bed415_0
- libstdcxx-ng=7.2.0=hdf63c60_3
- libxcb=1.12=hcd93eb1_4
- libxml2=2.9.7=h26e45fe_0
- lzo=2.10=h49e0be7_2
- markupsafe=1.0=py27h97b2822_1
- matplotlib=2.2.0=py27hbc4b006_0
- mistune=0.8.3=py27h14c3975_1
- mkl=2018.0.1=h19d6760_4
- nbconvert=5.3.1=py27he041f76_0
- nbformat=4.4.0=py27hed7f2b2_0
- ncurses=6.0=h9df7e31_2
- notebook=5.4.1=py27_0
- numexpr=2.6.4=py27hd318778_0
- numpy=1.14.2=py27hdbf6ddf_0
- openssl=1.0.2n=hb7f436b_0
- pandoc=1.19.2.1=hea2e7c5_1
- pandocfilters=1.4.2=py27h428e1e5_1
- pathlib2=2.3.2=py27_0
- pcre=8.41=hc27e229_1
- pexpect=4.5.0=py27_0
- pickleshare=0.7.4=py27h09770e1_0
- pip=9.0.1=py27_5
- prompt_toolkit=1.0.15=py27h1b593e1_0
- ptyprocess=0.5.2=py27h4ccb14c_0
- pygments=2.2.0=py27h4a8b6f5_0
- pyparsing=2.2.0=py27hf1513f8_1
- pyqt=5.6.0=py27h4b1e83c_5
- pytables=3.4.2=py27h1f7bffc_2
- python=2.7.14=h1571d57_30
- python-dateutil=2.7.0=py27_0
- pytz=2018.3=py27_0
- pyyaml=3.12=py27h2d70dd7_1
- pyzmq=17.0.0=py27h14c3975_1
- qt=5.6.2=hd25b39d_14
- qtconsole=4.3.1=py27hc444b0d_0
- readline=7.0=ha6073c6_4
- scandir=1.7=py27h14c3975_0
- send2trash=1.5.0=py27_0
- setuptools=38.5.1=py27_0
- simplegeneric=0.8.1=py27_2
- singledispatch=3.4.0.3=py27h9bcb476_0
- sip=4.18.1=py27he9ba0ab_2
- six=1.11.0=py27h5f960f1_1
- sqlite=3.22.0=h1bed415_0
- subprocess32=3.2.7=py27h373dbce_0
- terminado=0.8.1=py27_1
- testpath=0.3.1=py27hc38d2c4_0
- tk=8.6.7=hc745277_3
- tornado=5.0=py27_0
- traitlets=4.3.2=py27hd6ce930_0
- wcwidth=0.1.7=py27h9e3e1ab_0
- webencodings=0.5.1=py27hff10b21_1
- wheel=0.30.0=py27h2bc6bb2_1
- widgetsnbextension=3.2.1=py27_0
- xz=5.2.3=h55aa19d_2
- yaml=0.1.7=had09818_2
- zeromq=4.2.5=h439df22_0
- zlib=1.2.11=ha838bed_2
- pip:
  - absl-py==0.2.0
  - backports-abc==0.5
  - backports.functools-lru-cache==1.5
  - backports.shutil-get-terminal-size==1.0.0
  - backports.weakref==1.0.post1
  - chardet==3.0.4
  - funcsigs==1.0.2
  - future==0.16.0
  - gym==0.1.0
  - h5py==2.7.1
  - idna==2.6
  - imageio==2.3.0
  - ipython-genutils==0.2.0
  - jsanimation==0.1
  - jupyter-client==5.2.3
  - jupyter-console==5.2.0
  - jupyter-core==4.4.0
  - markdown==2.6.11
  - mock==2.0.0
  - mujoco-py==0.4.0
  - pandas==0.22.0
  - pbr==4.0.2
  - pillow==5.1.0
  - prompt-toolkit==1.0.15
  - protobuf==3.5.2.post1
  - pyglet==1.3.2
  - pyopengl==3.1.0
  - requests==2.18.4
  - scikit-learn==0.19.1
  - scipy==0.17.0
  - sklearn==0.0
  - tables==3.4.2
  - theano==0.8.2
  - urllib3==1.22
  - werkzeug==0.14.1
```
    