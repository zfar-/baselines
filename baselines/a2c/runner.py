from __future__ import division
import numpy as np
from baselines.a2c.utils import discount_with_dones
from baselines.common.runners import AbstractEnvRunner
from baselines.common.constants import constants
from baselines.common.mpi_moments import mpi_moments
from baselines.common.running_mean_std import RunningMeanStd




class Runner(AbstractEnvRunner):
    """
    We use this class to generate batches of experiences

    __init__:
    - Initialize the runner

    run():
    - Make a mini batch of experiences
    """
    def __init__(self, env, model, icm, curiosity, nsteps=5, gamma=0.99):
        super().__init__(env=env, model=model, icm=icm, nsteps=nsteps)
        self.gamma = gamma
        self.icm=icm
        self.curiosity = curiosity
        self.batch_action_shape = [x if x is not None else -1 for x in model.train_model.action.shape.as_list()]
        self.ob_dtype = model.train_model.X.dtype.as_numpy_dtype
        self.rff = RewardForwardFilter(self.gamma)
        self.rff_rms = RunningMeanStd()

    def run(self):
        # curiosity = True
        # curiosity = False

        # We initialize the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_next_states = [],[],[],[],[],[]
        mb_states = self.states
        icm_testing_rewards = []
        for n in range(self.nsteps):
            # Given observations, take action and value (V(s))
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, states, _ = self.model.step(self.obs, S=self.states, M=self.dones)

            # Append the experiences
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)

            if self.curiosity == True:
                icm_states=self.obs

            # Take actions in env and look the results
            obs, rewards, dones, _ = self.env.step(actions)
            # print("received Rewards from step function ")

            # print("received Rewards ",rewards)
            if self.curiosity == True:
                icm_next_states = obs

                icm_rewards = self.icm.calculate_intrinsic_reward(icm_states,icm_next_states,actions)
                # print("shape of icm rewards ",np.shape(icm_rewards))
                icm_testing_rewards.append(icm_rewards)
                # icm_rewards = [icm_rewards] * len(rewards)

                # icm_rewards = icm_rewards * 2
                # print("intrinsic Reward : ",icm_rewards)

                # icm_rewards = np.clip(icm_rewards,-constants['REWARD_CLIP'], constants['REWARD_CLIP'])
            
                # print("icm _ rewards : ",icm_rewards)
            

                
                # rewards = icm_rewards  + rewards
                # print("Rewards icm {} , commulative reward {} ".format(icm_rewards , rewards))
                
                # rewards = np.clip(rewards,-constants['REWARD_CLIP'], +constants['REWARD_CLIP'])
                # print("icm rewards ", rewards)
            
                # print("calculated rewards ",rewards)
                

            mb_next_states.append(np.copy(obs))
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
            self.obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)

        # Batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.ob_dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_next_states = np.asarray(mb_next_states , dtype=self.ob_dtype).swapaxes(1,0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        # > testing mean std of rewards 
        if self.curiosity:
            icm_testing_rewards = np.asarray(icm_testing_rewards, dtype=np.float32).swapaxes(1, 0)
            # print("Icm rewards" ,icm_testing_rewards)
        # > testing mean std of rewards 
        mb_actions = np.asarray(mb_actions, dtype=self.model.train_model.action.dtype.name).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]

        # > passing reward to reward forward filter 

        # print("Merged things obs {} rewards {} actions {} dones {}".
            # format(np.shape(mb_obs) , np.shape(mb_rewards) , np.shape(mb_actions) , np.shape(mb_dones)))

        # >
        # rffs = np.array([self.rff.update(rew) for rew in mb_rewards.T])

        if self.curiosity == True :
            rffs = np.array([self.rff.update(rew) for rew in icm_testing_rewards.T])
            rffs_mean, rffs_std, rffs_count = mpi_moments(rffs.ravel())
            self.rff_rms.update_from_moments(rffs_mean, rffs_std ** 2, rffs_count)
            rews = icm_testing_rewards / np.sqrt(self.rff_rms.var)

            # mb_rewards = rews

            mb_rewards = rews # mb_rewards + rews

            # now clipping the reward (-1,1)

            # mb_rewards = np.clip(mb_rewards,-constants['REWARD_CLIP'], constants['REWARD_CLIP'])
            # print(mb_rewards)
            # print(" shape of normalized reward ", np.shape(rews))


            # icm_testing_rewards = (icm_testing_rewards >  rffs_mean).astype(np.float32)
            # np.place(icm_testing_rewards, icm_testing_rewards > 0, 0.2)
    
        # np.interp(icm_testing_rewards.ravel() , (rffs_mean+)  , ())
        
        # icm_testing_rewards = icm_testing_rewards.ravel()
        # print("\n\nIcm Rewards : ",icm_testing_rewards)
        
        # print(" icm testing rewards ")
        # print("icm testing reward : mean {} , std {} , division {} ".format(rffs_mean , rffs_std , ((rffs_mean + rffs_std)/2 ) ) )

        # print("ICM testing rewards " , icm_testing_rewards)

        # icm_testing_rewards[icm_testing_rewards > rffs_mean] = 0.5
        # icm_testing_rewards[icm_testing_rewards < rffs_mean] = 0
        # icm_testing_rewards[icm_testing_rewards < rffs_mean] = 0
        # print("icm rewards ", icm_testing_rewards)

            # mb_rewards = icm_testing_rewards + mb_rewards

        # print( mb_rewards)
        # mb_rewards = mb_rewards[mb_rewards > 1]
        # mb_rewards = [1 if mb_rewards[mb_rewards >1 ] else 1]
        # mb_rewards[mb_rewards > 1] = 1  
        # mask = mb_rewards[((icm_testing_rewards + mb_rewards ) % 2) == 0]

        # print("Mask ",mask)
        # mb_rewards[mask == 0] = 1 

        # print("Mb reward ",mb_rewards )

        # print("Icm Rewards : ",icm_testing_rewards)
        # self.rff_rms.update_from_moments(rffs_mean, rffs_std ** 2, rffs_count)
        # rews = mb_rewards / np.sqrt(self.rff_rms.var)
        # >

        # print("update : rffs_mean {} , rffs_std {} , rffs_count {} ".format(
                # np.shape(rffs_mean),np.shape(rffs_std),np.shape(rffs_count)))

        # print(" update :  final rews {} rff_rms.var {} ".format(
                # rews , np.shape(self.rff_rms.var)))



        # print(">> the shape of rffs testing ", np.shape(rffs))

        # mb_rewards_copy = mb_rewards
        
        # if self.curiosity == True :
        #     if self.gamma > 0.0:
        #         # Discount/bootstrap off value fn
        #         last_values = self.model.value(self.obs, S=self.states, M=self.dones).tolist()
        #         for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
        #             rewards = rewards.tolist()
        #             dones = dones.tolist()
        #             # if dones[-1] == 0:
        #             rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
        #             # else:
        #             # rewards = discount_with_dones(rewards, dones, self.gamma)

        #             mb_rewards[n] = rewards
        # else :    
        # # print(" Before discount_with_dones ")
        # # print("Rewards " , mb_rewards)

        # # print("Before rewards and values ")
        # # print("Reward {} values {} ".format(mb_rewards , mb_values))
        #     if self.gamma > 0.0:
        #         # Discount/bootstrap off value fn
        #         last_values = self.model.value(self.obs, S=self.states, M=self.dones).tolist()
        #         for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
        #             rewards = rewards.tolist()
        #             dones = dones.tolist()
        #             if dones[-1] == 0:
        #                 rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
        #             else:
        #                 rewards = discount_with_dones(rewards, dones, self.gamma)

        #             mb_rewards[n] = rewards




        # print(" After discount_with_dones ")
        # print("Orgnal discounterd Rewards " , np.shape(mb_rewards))

        # rffs_mean, rffs_std, rffs_count = mpi_moments(mb_rewards.ravel())
        # self.rff_rms.update_from_moments(rffs_mean, rffs_std ** 2, rffs_count)
        # mb_rewards = mb_rewards_copy / np.sqrt(self.rff_rms.var)



        mb_actions = mb_actions.reshape(self.batch_action_shape)

        mb_rewards = mb_rewards.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        
        if self.curiosity == True :
            mb_rews_icm = rews.flatten()

        # mb_new_updated_reward = mb_rews_icm + mb_rewards

        # print("New udpated rewards ",mb_new_updated_reward)

        # rffs_mean, rffs_std, rffs_count = mpi_moments(mb_new_updated_reward.ravel())
        # self.rff_rms.update_from_moments(rffs_mean, rffs_std ** 2, rffs_count)
        # rews = mb_new_updated_reward / np.sqrt(self.rff_rms.var)

        # print("After normalized",rews)



        # mb_new_rew = rews.flatten()

        # print("Flatten rewards and values ")
        # print("Reward {} ".format(mb_rewards ))

        # print("Merged things after obs {} rewards {} actions {} masks {}".
            # format(np.shape(mb_obs) , np.shape(mb_rewards) , np.shape(mb_actions) , np.shape(mb_masks)))

        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, mb_next_states # , mb_rews_icm, mb_new_updated_reward #, mb_new_rew




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

