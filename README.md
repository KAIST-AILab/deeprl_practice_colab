# Deep Reinforcement Learning Practice with Google Colab

## Day 1
- 
  - Trust region policy optimization [[Schulman et al. ICML 2015]](http://proceedings.mlr.press/v37/schulman15.pdf)
## Day 2
- DQN (?)
## Day 3
- MAML


## Preparation Progress
0. ~~Colab에서 anaconda env로 rllab 실행하는걸 마무리~~. ~~Anaconda 안 쓰고, CUDNN을 따로 깔아서 GPU로 rllab 코드를 돌릴 수 있 colab ipython notebook도 따로 만듬~~.
1. ~~learned parameter, average return 저장 옵션 어떻게 주는지 확인~~ (trpo_cartpole.py, trpo_swimmer.py)
2. ~~iteration  별로 average return plot (pyplot)  (trpo_cartpole.py, trpo_swimmer.py)~~
3. code 블록 의미 있게 나눠서 따로 실행 가능하게 colab에 써보기
colab에 rendering하는 건 교수님 다음 colab에 대한 지시가 있을 때까지 대기
4. distral, maml 분석. rllab 사용 가능한지 보기.. (내 기억에는 maml rllab으로 구현됐던거 같음)
5. env observation normalization function 수정 (거의 모든 알고리즘에 적용되기 때문에 수정해야 함)
(baselines, rllab trpo가 같은 성능을 보이게)
6. ~~gym environment에 video recording=True옵션 줘서 mp4 저장하는 것 시도해볼 것~~
    - 내 개인적인 생각: rllab코드 내부에 mp4를 저장하는 코드가 있는데 (저장경로:/rllab/data/local/experiment/experiment_data_.../gym_log/openaigym.episode...) 는이를 활용할 수 있으면 의외로 쉽게 끝날 수도
        - 교수님께서 보신 봐로는 colab에지서 충분히 랜더가 된다고 하심... 다른 사람들이 올린 블로그에 나온다고 함
        - Policy parameters saved by logger in batch_polopt.py
        - Training process resumed by : rllab/scripts/resume_training.py.  The code uses `def load_params()` defined at `rllab/viskit/core.py` which loads `params.json`
        - In `rllab/viskit/core.py`, `to_json()` defined which is used for saving exp result and params (maybe... 저장경로: mp4파일과 비슷한 경로인데 openai~ dir 들어가기 전까).
        
        - `AverageReturn` computed by `np.mean(undiscounted_returns)` at `rllab/sampler/base.py`
        
        - `progress.csv` saved at `data` dir should contain info on AverageReturn. But there is nothing saved. Suitable option should be given to write those info (maybe).
        
        - codes in `score.py`, `visualize_hyperopt_results.ipynb` could be used.
        
    - 형주: 콜랩에서 한빛형 코드를 (제 콜랩 드라이브에서) run_experiment_lite를 씌워서 돌려봤는데, 딱히 에러가 나지도 않고 그냥 멈춥니다. 
    /rllab/scripts/run_experiment_lite.py 실행해주는
    /rllab/rllab/misc/instrument.py 코드의
    run_experiment_lite 함수 안에서
    서브 프로세스를 콜하는 부분:
    subprocess.call(
       command, shell=True, env=dict(os.environ, **env)
    )
    에서 죽는것 같습니다.
        - 수정된 파일 https://colab.research.google.com/drive/1Pfhane7wLW7jr2jyqhuGk65ZShuYNEyD#scrollTo=_KwYCkcKiV2N
    
        -  pip install box2d-py mako==1.0.7 Pygame  인스톨파일에 추가
    
    
    
    