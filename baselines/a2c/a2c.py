import time
import functools
import tensorflow as tf

from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common import tf_util
from baselines.common.policies import build_policy


from baselines.a2c.utils import Scheduler, find_trainable_variables
from baselines.a2c.runner import Runner

from baselines.common.ICM import ICM
from baselines.a2c.utils import get_mean_and_std

from tensorflow import losses
import numpy as np

class Model(object):

    """
    We use this class to :
        __init__:
        - Creates the step_model
        - Creates the train_model

        train():
        - Make the training part (feedforward and retropropagation of gradients)

        save/load():
        - Save load the model
    """
    def __init__(self, policy, env, nsteps, icm,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):

        sess = tf_util.get_session()
        nenvs = env.num_envs
        nbatch = nenvs*nsteps

        print("This is Icm in Model Init function " , type(icm))


        with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):
            # step_model is used for sampling
            step_model = policy(nenvs, 1, sess)

            # train_model is used to train our network
            train_model = policy(nbatch, nsteps, sess)

        A = tf.placeholder(train_model.action.dtype, train_model.action.shape)
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        # Calculate the loss
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Policy loss
        neglogpac = train_model.pd.neglogp(A)
        # L = A(s,a) * -logpi(a|s)
        pg_loss = tf.reduce_mean(ADV * neglogpac)

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # Value loss
        vf_loss = losses.mean_squared_error(tf.squeeze(train_model.vf), R)

        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

        # Update parameters using loss
        # 1. Get the model parameters
        params = find_trainable_variables("a2c_model")

        # 2. Calculate the gradients
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))

        if icm is not None :

            grads = grads + icm.pred_grads_and_vars
            # print("Gradients added ")
            # print("independetly there shape were a2c : {} icm :{} and together {} ".format(np.shape(grads),np.shape(icm.pred_grads_and_vars),
                # np.shape(grads_and_vars)))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        # 3. Make op for one policy and value update step of A2C
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)

        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values , next_obs ) :
        #, icm_rewards,cumulative_dicounted_icm): #, new_rew):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # rewards = R + yV(s')
            # print(" icm called in train function ", type(icm))
            advs = rewards - values



            # print("Now the advantage ", advs )

            # icm_adv = icm_rewards - values
            # m , s = get_mean_and_std(icm_adv)

            # > adv Normaliztion
            m , s = get_mean_and_std(advs)
            advs = (advs - m) / (s + 1e-7)



            # advs = (icm_adv - m) / (s + 1e-7)


            # icm_adv = (icm_adv - icm_adv.mean()) / (  + 1e-7) 
            # print("icm advantage ", icm_adv)


            # advs = new_rew - values
            # print("Advantage :", advs)
            # print("On train shapes are  ")
            # print(" obs {} states {} rewards {} masks {} actions {} values {} ".
                # format(np.shape(obs) , np.shape(states) , np.shape(rewards) , np.shape(masks) ,np.shape(actions) ,
                # np.shape(values) ))
            # print("Received Advantage {} rewards {} values {}".format(
                # advs , rewards , values) )

           
            # print("advantage reward and values shape ")
            # print("advs {} , rewards shape {} , values {}".format(np.shape(advs) , np.shape(rewards) , np.shape(values)))

            for step in range(len(obs)):
                cur_lr = lr.value()

            if icm is None :

                td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            else :
                # print("curiosity Td Map ")
                print(" obs {} , next obs {} , actions  {} ".format(np.shape(obs) , np.shape(next_obs),
                    np.shape(actions)))
                td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr , 
                icm.state_:obs, icm.next_state_ : next_obs , icm.action_ : actions }# , icm.R :rewards }



            if icm is None :
                if states is not None:
                    td_map[train_model.S] = states
                    td_map[train_model.M] = masks
                
                policy_loss, value_loss, policy_entropy, _ = sess.run(
                    [pg_loss, vf_loss, entropy, _train],
                    td_map
                )
                return policy_loss, value_loss, policy_entropy
            else :
                if states is not None:
                    td_map[train_model.S] = states
                    td_map[train_model.M] = masks
                policy_loss, value_loss, policy_entropy,forward_loss , inverse_loss , icm_loss, _ = sess.run(
                    [pg_loss, vf_loss, entropy, icm.forw_loss , icm.inv_loss, icm.icm_loss ,_train],
                    td_map

                )
                return policy_loss, value_loss, policy_entropy,forward_loss , inverse_loss , icm_loss, advs



        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = functools.partial(tf_util.save_variables, sess=sess)
        self.load = functools.partial(tf_util.load_variables, sess=sess)
        tf.global_variables_initializer().run(session=sess)


def learn(
    network,
    env,
    seed=None,
    curiosity=False,
    nsteps=5,
    total_timesteps=int(80e6),
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    lr=7e-4,
    lrschedule='linear',
    epsilon=1e-5,
    alpha=0.99,
    gamma=0.99,
    log_interval=100,
    load_path=None,
    **network_kwargs):

    '''
    Main entrypoint for A2C algorithm. Train a policy with given network architecture on a given environment using a2c algorithm.

    Parameters:
    -----------

    network:            policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                        specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                        tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                        neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                        See baselines.common/policies.py/lstm for more details on using recurrent nets in policies


    env:                RL environment. Should implement interface similar to VecEnv (baselines.common/vec_env) or be wrapped with DummyVecEnv (baselines.common/vec_env/dummy_vec_env.py)


    seed:               seed to make random number sequence in the alorightm reproducible. By default is None which means seed from system noise generator (not reproducible)

    nsteps:             int, number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                        nenv is number of environment copies simulated in parallel)

    total_timesteps:    int, total number of timesteps to train on (default: 80M)

    vf_coef:            float, coefficient in front of value function loss in the total loss function (default: 0.5)

    ent_coef:           float, coeffictiant in front of the policy entropy in the total loss function (default: 0.01)

    max_gradient_norm:  float, gradient is clipped to have global L2 norm no more than this value (default: 0.5)

    lr:                 float, learning rate for RMSProp (current implementation has RMSProp hardcoded in) (default: 7e-4)

    lrschedule:         schedule of learning rate. Can be 'linear', 'constant', or a function [0..1] -> [0..1] that takes fraction of the training progress as input and
                        returns fraction of the learning rate (specified as lr) as output

    epsilon:            float, RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update) (default: 1e-5)

    alpha:              float, RMSProp decay parameter (default: 0.99)

    gamma:              float, reward discounting parameter (default: 0.99)

    log_interval:       int, specifies how frequently the logs are printed out (default: 100)

    **network_kwargs:   keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                        For instance, 'mlp' network architecture has arguments num_hidden and num_layers.

    '''
    # curiosity = True
    # curiosity = False



    set_global_seeds(seed)

    # Get the nb of env
    nenvs = env.num_envs
    policy = build_policy(env, network, **network_kwargs)

    temp_ob_space = env.observation_space
    temp_ac_space = env.action_space


    temp_nbatch = nenvs * nsteps
    temp_nbatch_train = temp_nbatch 


    # Instantiate the model object (that creates step_model and train_model)
    if curiosity == False :
        model = Model(policy=policy, env=env, nsteps=nsteps, icm=None ,ent_coef=ent_coef, vf_coef=vf_coef,
            max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
    else :
        print("Called curiosity model")
        make_icm = lambda: ICM(ob_space = temp_ob_space, ac_space = temp_ac_space, max_grad_norm = max_grad_norm, beta = 0.2, icm_lr_scale = 0.1 )
        icm = make_icm()

        model = Model(policy=policy, env=env, nsteps=nsteps, icm=icm , ent_coef=ent_coef, vf_coef=vf_coef,
            max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)

        


    if load_path is not None:
        model.load(load_path)

    # Instantiate the runner object
    if curiosity == False:
        runner = Runner(env, model, nsteps=nsteps, curiosity=False ,icm=None, gamma=gamma)
    else :
        print("Called curiosity Runner")
        runner = Runner(env, model, nsteps=nsteps, curiosity=curiosity ,icm=icm, gamma=gamma)



    # Calculate the batch_size
    nbatch = nenvs*nsteps

    # Start total timer
    tstart = time.time()

    for update in range(1, total_timesteps//nbatch+1):
        # Get mini batch of experiences
        # print("Update step : ",update)
        obs, states, rewards, masks, actions, values, next_ob = runner.run() # ,icm_rewards,cumulative_dicounted_icm = runner.run()

        # > now here we will do the reward normalization 

        if curiosity == False :

           policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values,next_obs=None)
        else :
            policy_loss, value_loss, policy_entropy,forwardLoss , inverseLoss , icm_loss , advs = model.train(obs, states, rewards, masks, actions, values , next_ob)#, icm_rewards,cumulative_dicounted_icm)

            # print("Shape of ")
            # print( "policy_loss {}, value_loss {}, policy_entropy {},forwardLoss {} , inverseLoss {}, icm_loss {}".
                # format(np.shape(policy_loss) , np.shape(value_loss) , np.shape(policy_entropy) , np.shape(forwardLoss) , np.shape(inverseLoss), np.shape(icm_loss)))


        nseconds = time.time()-tstart

        # print("icm loss :" , np.mean(icm_loss))

        # Calculate the fps (frame per second)
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            if curiosity == True :
                # logger.record_tabular("forwardLoss", float(forwardLoss))
                # logger.record_tabular("inverseLoss", float(inverseLoss))
                logger.record_tabular("icm Loss", float(icm_loss))
                logger.record_tabular("Advantage" , np.mean(advs))
            

            logger.record_tabular("explained_variance", float(ev))
            logger.dump_tabular()
    return model

