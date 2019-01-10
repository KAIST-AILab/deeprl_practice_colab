# Day 3: Transfer Learning (Meta Learning) (Thinking of doing practice on GAIL with the paper author's code)
## Contents
### Used algorithm
Model-Agnostic Meta-Learning (MAML) [[Finn et al. ICML 2017]](http://proceedings.mlr.press/v70/finn17a.html).
### Environments
1. Classical Control: Cartpole
2. Mujoco Environment: Swimmer

### Notes
- Examples located at .../rllab/maml_exmaples
- Algorithms located at ../rllab/sandbox/rocky/tf/algos
- File run (?) order for MAML-TRPO:
    1. Run experiment with the wrapper (when used): run_experiment_lite.py 
    2. Run batch maml policy optimization algo (TRPO inherits this class and uses the methods in the class): batch_maml_polopt.py
    3. Make vectorized envs: sandbox/rocky/tf/samplers/vectorized_sampler.py
    4. Reset env (in process of vectorizing envs): sandbox/rocky/tf/envs/vec_env_executor.py
    5. rllab/rllab/envs/proxy_env.py