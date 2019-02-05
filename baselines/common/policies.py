import tensorflow as tf
from baselines.common import tf_util
from baselines.a2c.utils import fc, fcNoisy
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.tf_util import adjust_shape
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.models import get_network_builder

import gym


class PolicyWithValue(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, env, observations, latent, estimate_q=False, vf_latent=None, sess=None, **tensors):
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.X = observations
        self.state = tf.constant([])
        self.initial_state = None
        self.__dict__.update(tensors)

        vf_latent = vf_latent if vf_latent is not None else latent

        vf_latent = tf.layers.flatten(vf_latent)
        latent = tf.layers.flatten(latent)

        # >
        # action noise placeholders 
        self.noise=tf.placeholder(tf.float32, name="noise")
        self.newbie=tf.placeholder(tf.float32, name="newbie")
        self.Sigma=tf.placeholder(tf.float32, name="Sigma")
        # >

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(env.action_space)

        # self.pd, self.pi = self.pdtype.pdfromlatent(latent, init_scale=0.01)
        # > Noise pdffrom latent
        self.pd, self.pi = self.pdtype.pdfromlatent(latent, init_scale=0.01,Newbie=self.newbie,Noise=self.noise,sigma=self.Sigma)
        # >

        # Take an action
        self.action = self.pd.sample()

        # Calculate the neg log of our probability
        self.neglogp = self.pd.neglogp(self.action)
        self.sess = sess or tf.get_default_session()

        if estimate_q:
            assert isinstance(env.action_space, gym.spaces.Discrete)
            self.q = fc(vf_latent, 'q', env.action_space.n)
            self.vf = self.q
        else:
            # > action noise 
            self.vf = fcNoisy(vf_latent, 'vf', 1,newbie=self.newbie,noise=self.noise,sigma=self.Sigma)
            # self.vf = fc(vf_latent, 'vf', 1)
            self.vf = self.vf[:,0]


    # > three function of action nosie 
    # > 1. _evaluate 
    # > 2. step 
    # > 3. value
    def _evaluate(self, variables, observation,Noise=0.0,Newbie=0.0,sigma=0.0, **extra_feed):
        sess = self.sess or tf.get_default_session()
        feed_dict = {self.X: adjust_shape(self.X, observation),
        self.noise:Noise,
        self.newbie:Newbie,
        self.Sigma:sigma}

        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def step(self, observation,Noise=0.0,Newbie=0.0,sigma=0.0, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """

        a, v, state, neglogp = self._evaluate([self.action, self.vf, self.state, self.neglogp], observation,Noise,Newbie,sigma, **extra_feed)
        if state.size == 0:
            state = None
        return a, v, state, neglogp

    def value(self, ob,Noise=0.0,Newbie=0.0,sigma=0.0, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        return self._evaluate(self.vf, ob,Noise,Newbie,sigma, *args, **kwargs)

    # def _evaluate(self, variables, observation, **extra_feed):
    #     sess = self.sess
    #     feed_dict = {self.X: adjust_shape(self.X, observation)}
    #     for inpt_name, data in extra_feed.items():
    #         if inpt_name in self.__dict__.keys():
    #             inpt = self.__dict__[inpt_name]
    #             if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
    #                 feed_dict[inpt] = adjust_shape(inpt, data)

    #     return sess.run(variables, feed_dict)

    # def step(self, observation, **extra_feed):
    #     """
    #     Compute next action(s) given the observation(s)

    #     Parameters:
    #     ----------

    #     observation     observation data (either single or a batch)

    #     **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

    #     Returns:
    #     -------
    #     (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
    #     """

    #     a, v, state, neglogp = self._evaluate([self.action, self.vf, self.state, self.neglogp], observation, **extra_feed)
    #     if state.size == 0:
    #         state = None
    #     return a, v, state, neglogp

    # def value(self, ob, *args, **kwargs):
    #     """
    #     Compute value estimate(s) given the observation(s)

    #     Parameters:
    #     ----------

    #     observation     observation data (either single or a batch)

    #     **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

    #     Returns:
    #     -------
    #     value estimate
    #     """
    #     return self._evaluate(self.vf, ob, *args, **kwargs)

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)




def build_policy(env, policy_network, value_network=None,  normalize_observations=False, estimate_q=False, **policy_kwargs):
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(**policy_kwargs)

        print("Building Policy with observation ",normalize_observations )
    def policy_fn(nbatch=None, nsteps=None, sess=None, observ_placeholder=None):
        ob_space = env.observation_space

        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)

        extra_tensors = {}

        print("Policies observation normalize_observations")
        # print("Its called here with type ", X.dtype)

        if normalize_observations: #  and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(tf.to_float(X))
            extra_tensors['rms'] = rms
        else:
            encoded_x = X

        encoded_x = encode_observation(ob_space, encoded_x)

        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            policy_latent = policy_network(encoded_x)
            if isinstance(policy_latent, tuple):
                policy_latent, recurrent_tensors = policy_latent

                if recurrent_tensors is not None:
                    # recurrent architecture, need a few more steps
                    nenv = nbatch // nsteps
                    assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
                    policy_latent, recurrent_tensors = policy_network(encoded_x, nenv)
                    extra_tensors.update(recurrent_tensors)


        _v_net = value_network

        if _v_net is None or _v_net == 'shared':
            vf_latent = policy_latent
        else:
            if _v_net == 'copy':
                _v_net = policy_network
            else:
                assert callable(_v_net)

            with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
                # TODO recurrent architectures are not supported with value_network=copy yet
                vf_latent = _v_net(encoded_x)

        policy = PolicyWithValue(
            env=env,
            observations=X,
            latent=policy_latent,
            vf_latent=vf_latent,
            sess=sess,
            estimate_q=estimate_q,
            **extra_tensors
        )
        return policy

    return policy_fn


def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    print(" Calcualted obv mean {} and std {} ".format(rms.mean,rms.std))
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms

