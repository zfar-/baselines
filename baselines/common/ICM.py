import tensorflow as tf
import numpy as np
from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
from baselines.common.tf_util import get_session, save_variables, load_variables

from mpi4py import MPI
from baselines.common.tf_util import initialize
from baselines.common.mpi_util import sync_from_root
from baselines.common.constants import constants

# Helper function to build convnets
def conv2d(inputs, filters, kernel_size, strides, padding):
        return tf.layers.conv2d(inputs = inputs,
                           filters = filters,
                            kernel_size = (kernel_size, kernel_size),
                            strides = strides,
                            padding = padding)


class ICM(object):
    def __init__(self, ob_space, ac_space, max_grad_norm, beta, icm_lr_scale):

        sess = get_session()

        #TODO find a better way
        input_shape = [ob_space.shape[0], ob_space.shape[1], ob_space.shape[2]]

        # input_shape = ob_space
        print("ICM state Input shape ", np.shape(input_shape) , "  ", input_shape)
        self.action_shape = 36
            
        # Placeholders

        self.state_ = phi_state = tf.placeholder(tf.float32, [None, *input_shape], name="icm_state")
        self.next_state_ = phi_next_state =  tf.placeholder(tf.float32, [None, *input_shape], name="icm_next_state")
        self.action_ = action = tf.placeholder(tf.float32, [None], name="icm_action")
        # self.R = rewards = tf.placeholder(tf.float32, shape=[None], name="maxR")


        with tf.variable_scope('icm_model'):
            # Feature encoding
            # Aka pass state and next_state to create phi(state), phi(next_state)
            # state --> phi(state)
            print("Feature Encodding of phi state with shape :: ",self.state_)
            phi_state = self.feature_encoding(self.state_)

            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                # next_state to phi(next_state)
                phi_next_state = self.feature_encoding(self.next_state_)
            
            # INVERSE MODEL
            pred_actions_logits, pred_actions_prob = self.inverse_model(phi_state, phi_next_state)
            
            # FORWARD MODEL
            pred_phi_next_state = self.forward_model(action, phi_state)


        # CALCULATE THE ICM LOSS
        # Inverse Loss LI
        # We calculate the cross entropy between our ât and at
        # Squeeze the labels (required)
        labels = tf.cast(action, tf.int32)

        print("prediction pred_actions_logits")

        self.inv_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_actions_logits, labels=labels),name="inverse_loss")

        # Foward Loss
        # LF = 1/2 || pred_phi_next_state - phi_next_state ||
        # TODO 0.5 * ?
        self.forw_loss_axis = tf.reduce_mean(tf.square(tf.subtract(pred_phi_next_state, phi_next_state)) , axis=-1 , name="forward_loss_axis")

        self.forw_loss = tf.reduce_mean(tf.square(tf.subtract(pred_phi_next_state, phi_next_state))  , name="forward_loss")


        # Todo predictor lr scale ?
        # ICM_LOSS = [(1 - beta) * LI + beta * LF ] * Predictor_Lr_scale
        self.icm_loss = ((1-beta) * self.inv_loss + beta * self.forw_loss) * icm_lr_scale

        ####
        # self.icm_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        # print("ICM var list ::: " , self.icm_var_list)
        ####

        # 
        # if max_grad_norm is not None :
            # t_icm_grads , _ = tf.clip_by_global_norm(self.icm_loss, constants['GRAD_NORM_CLIP'] ) 
        # t_icm_grads_and_vars = list(zip(self.icm_loss , self.icm_var_list))
        # print("\n\n\nit works \n\n\n")
        # 


        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        self.icm_params = tf.trainable_variables('icm_model')  ## var_list same as 
        
        ## testing phase 
        self.predgrads  = tf.gradients(self.icm_loss, self.icm_params)
        self.predgrads , _ = tf.clip_by_global_norm(self.predgrads ,max_grad_norm )
        self.pred_grads_and_vars = list(zip(self.predgrads, self.icm_params))

        ## testing phase



        # print("\n\nTrainable variables \n ",icm_params)
        # # 2. Build our trainer
        self.icm_trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=1e-4, epsilon=1e-5)
        # # 3. Calculate the gradients
        icm_grads_and_var = self.icm_trainer.compute_gradients(self.icm_loss, self.icm_params)
        # # t_grads_and_var = tf.gradients()
        icm_grads, icm_var = zip(*icm_grads_and_var)

        if max_grad_norm is not None:
        #     # Clip the gradients (normalize)
            icm_grads, icm__grad_norm = tf.clip_by_global_norm(icm_grads, max_grad_norm)
        icm_grads_and_var= list(zip(icm_grads, icm_var))
        # # zip aggregate each gradient with parameters associated
        # # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        self._icm_train = self.icm_trainer.apply_gradients(icm_grads_and_var)



        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Initialize")
            initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        # print("GLOBAL VARIABLES", global_variables)
        sync_from_root(sess, global_variables) #pylint: disable=E1101        
    
    
    # We use batch normalization to do feature normalization as explained in the paper
    # using the universe head, 
    def feature_encoding(self, x):
        print("feature function called !!")
        x = tf.nn.elu(tf.layers.batch_normalization(conv2d(x, 8, 5, 4, "valid")))
        print(x)
        x = tf.nn.elu(tf.layers.batch_normalization(conv2d(x, 16, 3, 2, "valid")))
        print(x)
        x = tf.nn.elu(tf.layers.batch_normalization(conv2d(x, 32, 3, 2, "valid")))
        print(x)
        x = tf.nn.elu(tf.layers.batch_normalization(conv2d(x, 64, 3, 2, "valid")))
        print(x)
        x = tf.layers.flatten(x)
        x = tf.nn.elu(tf.contrib.layers.fully_connected(x, 256))

        return x



    # Inverse Model
    # Given phi(state) and phi(next_state) returns the predicted action ât
    """
    Parameters
    __________
    
    action:   The real action taken by our agent
    phi_state: The feature representation of our state generated by our feature_encoding function.
    phi_next_state: The feature representation of our next_state generated by our feature_encoding function.
    
    returns pred_actions_logits: the logits and pred_actions_prob: the probability distribution of our actions
    """
    def inverse_model(self, phi_state, phi_next_state):
        # Concatenate phi(st) and phi(st+1)
        icm_inv_concatenate = tf.concat([phi_state, phi_next_state], 1)
        icm_inv_fc1 = tf.nn.relu(tf.layers.dense(icm_inv_concatenate, 256))
        pred_actions_logits = tf.layers.dense(icm_inv_fc1, self.action_shape)
        pred_actions_prob = tf.nn.softmax(pred_actions_logits, dim=-1)
        
        return pred_actions_logits, pred_actions_prob
        
    # Foward Model
    # Given action and phi(st) must find pred_phi(st+1)
    """
    Parameters
    __________
    
    action:   The action taken by our agent
    phi_state: The feature representation of our state generated by our feature_encoding function.
    phi_next_state: The feature representation of our next_state generated by our feature_encoding function.
    
    returns pred_phi_next_state: The feature representation prediction of our next_state.
    """
    def forward_model(self, action, phi_state):
        # Concatenate phi_state and action
        action = tf.expand_dims(action, axis=1) # Expand dimension to be able to concatenate

        icm_forw_concatenate = tf.concat(axis=1, values=[phi_state, action])

        # FC
        icm_forw_fc1 = tf.layers.dense(icm_forw_concatenate, 256)

        # FC (size of phi_state [1] aka the width) # size of 288
        icm_forw_pred_next_state = tf.layers.dense(icm_forw_fc1, phi_state.get_shape()[1].value)

        return icm_forw_pred_next_state
        
    
    # Calculate intrinsic reward
    """
    Parameters
    __________
    
    phi_next_state: The feature representation of our next_state generated by our feature_encoding function.
    pred_phi_next_state:   The feature representation prediction of our next_state.
    
    
    returns intrinsic_reward: The intrinsic reward
    """
    def calculate_intrinsic_reward(self, state, next_state, action):
        # print("In the error function ")

        sess = tf.get_default_session()
        # print("passed states shape {} {} {} ".format(np.shape(state) , np.shape(next_state) , np.shape(action)))
        # passed states shape (2, 84, 84, 4) (2, 84, 84, 4) (2,) 
        # print("action : {} , type {}".format(np.shape(action) , type(action)))
        nenvs = np.shape(state)[0]
        # print("nenvs ",nenvs)
        # tmp = []
        # for i in range(nenvs) :
        #     ac = [action[i]]
        #     tmp.append(sess.run(self.forw_loss,
        #         {self.state_: np.expand_dims(state[i,:,:,:], axis=0), self.next_state_: np.expand_dims(next_state[i,:,:,:],axis=0), 
        #         self.action_:  ac } ) )
            # print(" shape passed i {}, state {} , next_state {} , action _type  {} , action {} ".
                # format(i, np.shape(np.expand_dims(state[i,:,:,:] , axis=0)) , np.shape(next_state[i,:,:,:]) , 
                    # type(np.array(action[i] )) , np.shape([action[i]]) ) )
        
        # tmp = np.concatenate([sess.run(self.forw_loss, 
            # {self.state_: np.expand_dims(state[i,:,:,:], axis=0), 
            # self.next_state_: np.expand_dims(next_state[i,:,:,:],axis=0), self.action_:  [action[i]]}) for i in range(nenvs)] , 0 )     
        # print("tmp : ", np.shape(tmp) )
        error = sess.run(self.forw_loss_axis, {self.state_: state, self.next_state_: next_state, self.action_: action})
        # print("orignal error  + error with axis -1 ")
        # print(list(zip(tmp,error)))
        # print("orignal Error ",error)
        # error = error * 0.5 #np.dot(error , 0.5)
        # print("Return error ",error)

        # Return intrinsic reward
        return error

    def train_curiosity_model(self, states , next_states , actions):# , rewards):
        sess = tf.get_default_session()
        feed = {self.state_: states , self.next_state_ : next_states , self.action_ : actions }#, self.R :rewards }

        return sess.run((self.forw_loss, self.inv_loss, self.icm_loss, self._icm_train), feed_dict = feed)
        # pass




"""



   Need implement train function

        
"""