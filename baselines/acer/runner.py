import numpy as np
from baselines.common.runners import AbstractEnvRunner
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from gym import spaces

from baselines.common.constants import constants
from baselines.common.mpi_moments import mpi_moments
from baselines.common.running_mean_std import RunningMeanStd

class Runner(AbstractEnvRunner):

    def __init__(self, env, model, nsteps, icm , gamma , curiosity):
        super().__init__(env=env, model=model, nsteps=nsteps , icm=icm)
        assert isinstance(env.action_space, spaces.Discrete), 'This ACER implementation works only with discrete action spaces!'
        assert isinstance(env, VecFrameStack)

        self.nact = env.action_space.n
        nenv = self.nenv
        self.nbatch = nenv * nsteps
        self.batch_ob_shape = (nenv*(nsteps+1),) + env.observation_space.shape

        self.obs = env.reset()
        self.obs_dtype = env.observation_space.dtype
        self.ac_dtype = env.action_space.dtype
        self.nstack = self.env.nstack
        self.nc = self.batch_ob_shape[-1] // self.nstack

        # >
        self.curiosity = curiosity
        if self.curiosity :
            self.rff = RewardForwardFilter(gamma)
            self.rff_rms = RunningMeanStd()


        # >


    def run(self):
        # enc_obs = np.split(self.obs, self.nstack, axis=3)  # so now list of obs steps
        enc_obs = np.split(self.env.stackedobs, self.env.nstack, axis=-1)
        mb_obs, mb_actions, mb_mus, mb_dones, mb_rewards, mb_next_states = [], [], [], [], [], []
        icm_testing_rewards = []

        for _ in range(self.nsteps):
            actions, mus, states = self.model._step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_mus.append(mus)
            mb_dones.append(self.dones)

            # >
            if self.curiosity :
                # print("3 icm here ")
                icm_states = self.obs
            # >

            obs, rewards, dones, _ = self.env.step(actions)
            # states information for statefull models like LSTM
            
            if self.curiosity :
                icm_next_states = obs 
                # print("Sent parameters for \n icm_states {} , icm_next_states {} , actions {}".format(
                    # icm_states.shape , icm_next_states.shape , actions.shape))
                icm_rewards =  self.icm.calculate_intrinsic_reward(icm_states,icm_next_states,actions)
                icm_testing_rewards.append(icm_rewards)

            mb_next_states.append(np.copy(obs)) # s_t+1

            self.states = states
            self.dones = dones
            self.obs = obs
            mb_rewards.append(rewards)
            enc_obs.append(obs[..., -self.nc:])

        mb_obs.append(np.copy(self.obs))
        mb_dones.append(self.dones)
        mb_next_states.append(np.copy(obs))
        
        icm_actions = mb_actions 

        # >

        if self.curiosity :
        #     # print("5 icm here ")
        #     icm_testing_rewards.append(rewards)
        
            icm_testing_rewards = np.array(icm_testing_rewards , dtype=np.float32).swapaxes(1, 0)

        # >




        enc_obs = np.asarray(enc_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_obs = np.asarray(mb_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=self.ac_dtype).swapaxes(1, 0)

        # >
        icm_actions.append(actions)
        icm_actions = np.asarray(icm_actions, dtype=self.ac_dtype).swapaxes(1, 0)
        
        mb_next_states = np.array(mb_next_states, dtype=self.obs_dtype).swapaxes(1,0)
        # >
        
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)

        if self.curiosity:  # r_e + discounted( r_i )
            rffs = np.array([self.rff.update(rew) for rew in icm_testing_rewards.T])
            rffs_mean, rffs_std, rffs_count = mpi_moments(rffs.ravel())
            self.rff_rms.update_from_moments(rffs_mean, rffs_std ** 2, rffs_count)
            rews = icm_testing_rewards / np.sqrt(self.rff_rms.var)

            mb_rewards = rews + mb_rewards


        mb_mus = np.asarray(mb_mus, dtype=np.float32).swapaxes(1, 0)

        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)

        mb_masks = mb_dones # Used for statefull models like LSTM's to mask state when done
        mb_dones = mb_dones[:, 1:] # Used for calculating returns. The dones array is now aligned with rewards

        # shapes are now [nenv, nsteps, []]
        # When pulling from buffer, arrays will now be reshaped in place, preventing a deep copy.

        # print("sent parameters \n mb_obs {} next_obs {} mb_actions {} mb_rewards {} , mb_icm_actions {} , icm_testing_rewards {} ".format( 
            # mb_obs.shape, mb_next_states.shape , mb_actions.shape, mb_rewards.shape , icm_actions.shape , icm_testing_rewards.shape) )



        return enc_obs, mb_obs, mb_actions, mb_rewards, mb_mus, mb_dones, mb_masks, mb_next_states, icm_actions



class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            

            # print("RewardForwardFilter , rews {}".format(rews))
            self.rewems = self.rewems * self.gamma + rews
            # print("RewardForwardFilter , self.rewems {} ".format(
            # self.rewems) )
        # print("RewardForwardFilter , self.rewems {} ".format(
            # self.rewems) )

        # print("RewardForwardFilter , rews {}".format(rews))
        return self.rewems
