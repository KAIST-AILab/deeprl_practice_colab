from rllab.algos.base import RLAlgorithm
from rllab.misc.overrides import overrides
from rllab.misc import special
from rllab.misc import ext
from rllab.sampler import parallel_sampler
from rllab.plotter import plotter
from functools import partial
import rllab.misc.logger as logger
import theano.tensor as TT
import pickle as pickle
import numpy as np
import pyprind
import lasagne
from collections import deque
from rllab.algos.ddpg import parse_update_method, SimpleReplayPool

def huber_loss(diffs, clip):
    return TT.mean(TT.switch(abs(diffs) < clip, 0.5 * TT.square(diffs), clip * (abs(diffs) - 0.5 * clip)))
    
    
class DQN(RLAlgorithm):
    """
    Deep Q Network
    """

    def __init__(
            self,
            env,
            policy,
            es,
            batch_size=32,
            n_steps=8000000,
            min_pool_size=1000000,
            replay_pool_size=1000000,
            discount=0.99,
            policy_update_method='adam',
            policy_learning_rate=1e-3,
            target_model_update=10000,
            n_updates_per_sample=1,
            train_epoch_interval=10000,
            max_path_length=np.inf,
            n_eval_samples=5,
            delta_clip=np.inf,
            include_horizon_terminal_transitions=False,
            plot=False,
            pause_for_plot=False):
        """
        :param env: Environment
        :param policy: Policy
        :param qf: Q function
        :param es: Exploration strategy
        :param batch_size: Number of samples for each minibatch.
        :param n_epochs: Number of epochs. Policy will be evaluated after each epoch.
        :param epoch_length: How many timesteps for each epoch.
        :param min_pool_size: Minimum size of the pool to start training.
        :param replay_pool_size: Size of the experience replay pool.
        :param discount: Discount factor for the cumulative return.
        :param max_path_length: Discount factor for the cumulative return.
        :param qf_weight_decay: Weight decay factor for parameters of the Q function.
        :param qf_update_method: Online optimization method for training Q function.
        :param qf_learning_rate: Learning rate for training Q function.
        :param policy_weight_decay: Weight decay factor for parameters of the policy.
        :param policy_update_method: Online optimization method for training the policy.
        :param policy_learning_rate: Learning rate for training the policy.
        :param n_eval_samples: Number of samples (timesteps) for evaluating the policy.
        :param soft_target_tau: Interpolation parameter for doing the soft target update.
        :param n_updates_per_sample: Number of Q function and policy updates per new sample obtained
        :param scale_reward: The scaling factor applied to the rewards when training
        :param include_horizon_terminal_transitions: whether to include transitions with terminal=True because the
        horizon was reached. This might make the Q value back up less stable for certain tasks.
        :param plot: Whether to visualize the policy performance after each train_epoch_interval.
        :param pause_for_plot: Whether to pause before continuing when plotting.
        :return:
        """
        self.env = env
        self.policy = policy
        self.es = es
        self.batch_size = batch_size
        self.min_pool_size = min_pool_size
        self.replay_pool_size = replay_pool_size
        self.discount = discount
        self.n_steps = n_steps
        
        self.policy_learning_rate = policy_learning_rate
        self.policy_update_method = parse_update_method(
                policy_update_method,
                learning_rate=policy_learning_rate,
        )
        
        assert target_model_update >= 0
        if target_model_update >= 1:
            self.target_model_update = int(target_model_update) # hard update every xx steps
        else:
            self.target_model_update = float(target_model_update) # soft update
            
        self.n_updates_per_sample = n_updates_per_sample
        self.train_epoch_interval = train_epoch_interval
        self.max_path_length = max_path_length
        self.n_eval_samples = n_eval_samples
        self.delta_clip = delta_clip
        
        self.include_horizon_terminal_transitions = include_horizon_terminal_transitions
        self.plot = plot
        self.pause_for_plot = pause_for_plot

        self.qf_loss_averages = []
        self.q_averages = []
        self.y_averages = []
        self.paths = []
        self.es_path_returns = []
        self.paths_samples_cnt = 0

        self.opt_info = None

    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.policy)
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    @overrides
    def train(self):
        # This seems like a rather sequential method
        pool = SimpleReplayPool(
            max_pool_size=int(self.replay_pool_size),
            observation_dim=self.env.observation_space.flat_dim,
            action_dim=self.env.action_space.flat_dim,
        )
            
        self.start_worker()

        self.init_opt()
        itr = 0
        path_length = 0
        path_return = 0
        terminal = False
        
        observation = self.env.reset()

        sample_policy = pickle.loads(pickle.dumps(self.policy))

        train_epoch = 0
        
        while train_epoch * self.train_epoch_interval < self.n_steps: 
            logger.push_prefix('step #%d | ' % (train_epoch * self.train_epoch_interval))
            logger.log("Training started")
            for train_epoch_step in pyprind.prog_bar(range(self.train_epoch_interval)):
                # Execute policy
                if terminal or path_length > self.max_path_length:
                    # Note that if the last step step ends an episode, the very
                    # last state and observation will be ignored and not added
                    # to the replay pool
                    # print('terminal! ' + str(itr))
                    observation = self.env.reset()
                    self.es.reset()
                    sample_policy.reset()
                    self.es_path_returns.append(path_return)
                    path_length = 0
                    path_return = 0
                action = self.es.get_action(itr, observation, policy=sample_policy)  # qf=qf)

                next_observation, reward, terminal, _ = self.env.step(action)
                path_length += 1
                path_return += reward

                
                pool.add_sample(self.env.observation_space.flatten(observation), 
                                self.env.action_space.flatten(action), 
                                reward, 
                                terminal) ## clipping?

                observation = next_observation

                if pool.size >= self.min_pool_size:
                    for update_itr in range(self.n_updates_per_sample):
                        # Train policy
                        batch = pool.random_batch(self.batch_size)
                        self.do_training(itr, batch)
                    sample_policy.set_param_values(self.policy.get_param_values())
                
                itr += 1

            logger.log("Training finished")
            if pool.size >= self.min_pool_size:
                self.evaluate(train_epoch * self.train_epoch_interval, pool)
                if self.n_eval_samples > 0: # we performed rollout!
                    observation = self.env.reset()
                params = self.get_epoch_snapshot(train_epoch * self.train_epoch_interval)
                logger.save_itr_params(train_epoch * self.train_epoch_interval, params)
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()
            train_epoch += 1
            if self.plot:
                self.update_plot()
                if self.pause_for_plot:
                    input("Plotting evaluation run: Press Enter to "
                              "continue...")
        self.env.terminate()
        self.policy.terminate()

    def init_opt(self):
        # First, create "target" policy and Q functions
        target_policy = pickle.loads(pickle.dumps(self.policy))

        # y need to be computed first
        obs = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
        )

        # The yi values are computed separately as above and then passed to
        # the training functions below
        action = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )
        yvar = TT.vector('ys')

        qval = self.policy.get_qval_sym(obs, action).flatten()

        # qf_loss = huber_loss(yvar - qval, self.delta_clip)
        qf_loss = TT.mean(TT.square(yvar - qval))
        qf_reg_loss = qf_loss # + qf_weight_decay_term
       
        qf_updates = self.policy_update_method(
            qf_reg_loss, 
            self.policy.get_params(trainable=True))

        f_train_qf = ext.compile_function(
            inputs=[yvar, obs, action],
            outputs=[qf_loss, qval],
            updates=qf_updates
        )
        
        # qval_test = self.policy.get_qval_sym_test(obs, action)
        
#         f_train_qf_test = ext.compile_function(
#             inputs=[yvar, obs, action],
#             outputs=qval_test
#         )

        self.opt_info = dict(
            f_train_qf=f_train_qf,
            # f_train_qf_test=f_train_qf_test,
            target_policy=target_policy,
        )

    def do_training(self, itr, batch):
        
        # obs, actions, next_obs should be all flattened
        obs, actions, rewards, next_obs, terminals = ext.extract(
            batch,
            "observations", "actions", "rewards", "next_observations",
            "terminals"
        )
        
        target_policy = self.opt_info["target_policy"]
        next_actions, next_actions_info = target_policy.get_actions(next_obs)
        next_qvals = next_actions_info['q'][range(self.batch_size),next_actions]
        
        if np.any(np.isnan(next_qvals)):
            print(itr)
            print(target_policy.get_param_values())
            print(next_qvals)
            assert False
        
        ys = rewards + (1. - terminals) * self.discount * next_qvals
        
        # print(max(rewards), min(rewards), max(terminals), min(terminals))
        
        f_train_qf = self.opt_info["f_train_qf"]  
        qf_loss, qval = f_train_qf(ys, obs, actions)
        
        if self.target_model_update >= 1 and itr % self.target_model_update == 0:
            target_policy.set_param_values(self.policy.get_param_values())
        elif self.target_model_update < 1:
            target_policy.set_param_values(
                target_policy.get_param_values() * (1.0 - self.target_model_update) + 
                self.policy.get_param_values() * self.target_model_update) 

        self.qf_loss_averages.append(qf_loss)
        self.q_averages.append(qval)
        self.y_averages.append(ys)

    def evaluate(self, epoch, pool):
        
        logger.record_tabular('Epoch', epoch)
        
        if self.n_eval_samples > 0:
            logger.log("Collecting samples for evaluation")
            paths = parallel_sampler.sample_paths(
                policy_params=self.policy.get_param_values(),
                max_samples=self.n_eval_samples,
                max_path_length=self.max_path_length,
            )

            average_discounted_return = np.mean(
                [special.discount_return(path["rewards"], self.discount) for path in paths]
            )

            returns = [sum(path["rewards"]) for path in paths]
            
            average_action = np.mean(np.square(np.concatenate(
                                    [path["actions"] for path in paths])))
            
            logger.record_tabular('AverageReturn', np.mean(returns))
            logger.record_tabular('StdReturn', np.std(returns))
            logger.record_tabular('MaxReturn', np.max(returns))
            logger.record_tabular('MinReturn', np.min(returns))
            
            logger.record_tabular('AverageDiscountedReturn', average_discounted_return)
            
            logger.record_tabular('AverageAction', average_action)
            
            self.env.log_diagnostics(paths)
            self.policy.log_diagnostics(paths)

        all_qs = np.concatenate(self.q_averages)
        all_ys = np.concatenate(self.y_averages)

        average_q_loss = np.mean(self.qf_loss_averages)
        

        policy_reg_param_norm = np.linalg.norm(
            self.policy.get_param_values(regularizable=True)
        )
#         qfun_reg_param_norm = np.linalg.norm(
#             self.qf.get_param_values(regularizable=True)
#         )
          
        if len(self.es_path_returns) > 0:
            logger.record_tabular('AverageEsReturn',
                                  np.mean(self.es_path_returns))
            logger.record_tabular('StdEsReturn',
                                  np.std(self.es_path_returns))
            logger.record_tabular('MaxEsReturn',
                                  np.max(self.es_path_returns))
            logger.record_tabular('MinEsReturn',
                                  np.min(self.es_path_returns))
            logger.record_tabular('NbEs', len(self.es_path_returns))
        
        logger.record_tabular('AverageQLoss', average_q_loss)
        logger.record_tabular('AverageQ', np.mean(all_qs))
        logger.record_tabular('AverageAbsQ', np.mean(np.abs(all_qs)))
        logger.record_tabular('AverageY', np.mean(all_ys))
        logger.record_tabular('AverageAbsY', np.mean(np.abs(all_ys)))
        logger.record_tabular('AverageAbsQYDiff',
                              np.mean(np.abs(all_qs - all_ys)))

        logger.record_tabular('PolicyRegParamNorm',
                              policy_reg_param_norm)

        self.qf_loss_averages = []

        self.q_averages = []
        self.y_averages = []
        self.es_path_returns = []

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)

    def get_epoch_snapshot(self, epoch):
        return dict(
            env=self.env,
            epoch=epoch,
            policy=self.policy,
            target_policy=self.opt_info["target_policy"],
            es=self.es,
        )
