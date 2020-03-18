from Main.AlphaZero.DistributedSelfPlay import Constants as C
from Main.AlphaZero.Oracle import GraphOptimizer, OracleCommands
from Main import Hyperparameters, MachineSpecificSettings

# from keras import backend as K
# import tensorflow as tf
import numpy as np

ORACLE_PIPE = None
K = None
tf = None


def runPredictionOracle(model, selfPlayPool, toOraclePipe, kerasBackend, tensorflow):
    global ORACLE_PIPE, K, tf
    ORACLE_PIPE = toOraclePipe
    K = kerasBackend
    tf = tensorflow
    _runNormalKerasOracle(model, selfPlayPool)

'''
    if (MachineSpecificSettings.IS_UNIX_MACHINE):
        _runOptimizedGraphOracle(model, selfPlayPool)
    else:
        _runNormalKerasOracle(model, selfPlayPool)
'''
    # _runUnbiasedOracle(selfPlayPool)


# ***** Main oracle loop that's called from of the pre-defined oracle. ******
def _oracleLoop(predictFunc, selfPlayPool):
    predictionEvalHistory = []
    amountOfGames = []

    while (True):
        message = ORACLE_PIPE.get()

        if (message[0] == OracleCommands.EVAL_NEW_DATA):
            _, amountOfStates, states, workerID = message
            assert (amountOfStates == len(states))
            predictions = predictFunc(states)

            # DEBUG STATS
            amountOfGames.append(amountOfStates)
            predictionEvalHistory.extend(predictions[0])

            outMsg = (OracleCommands.ORACLE_STATUS_RUNNING, len(predictions[0]), predictions)
            selfPlayPool.sendMessageToProc(workerID, outMsg)

        # If we set the ABORT_FLAG to True whilst the oracle is listening on the Q we will get stuck in an infinite read.
        # Therefore we also send a message to the oracle when the cycle is over
        elif (message[0] == C.LocalWorkerProtocol.SELF_PLAY_OVER):
            _flushAndAbortLocalWorkers(selfPlayPool)
            break


def _flushAndAbortLocalWorkers(selfPlayPool):
    abortMsg = (C.LocalWorkerProtocol.SELF_PLAY_OVER, [])
    selfPlayPool.broadcastMsg(abortMsg)


# ***** Prediction with a simple keras model... *****
def _runNormalKerasOracle(model, selfPlayPool):
    global NORMAL_MODEL
    NORMAL_MODEL = model
    _oracleLoop(_predictWithNormalModel, selfPlayPool)


NORMAL_MODEL = None


def _predictWithNormalModel(states):
    grads = K.gradients(NORMAL_MODEL.output,NORMAL_MODEL.input[1])
    #a = NORMAL_MODEL.predict([states,np.array([1, 1, 1, 1, 1, 1, 1])])
    sess = K.get_session()
    act = np.zeros((1,7))
    for i in range(5):
        grad,out = sess.run([grads,NORMAL_MODEL.output],feed_dict={NORMAL_MODEL.input[0]:np.array(states), NORMAL_MODEL.input[1]:np.array(act)})
        print(grad,out.size())
    return out,grad


# ***** Prediction with unbiased values, AKA fake prediction without any network... *****
UNBIASED_EVAL = None
UNBIASED_POLICY = None


# ***** Prediction with a simple keras model... *****
def _runUnbiasedOracle(selfPlayPool):
    global UNBIASED_EVAL, UNBIASED_POLICY
    UNBIASED_EVAL = [[np.random.random()] for i in range(Hyperparameters.AMOUNT_OF_GAMES_PER_WORKER)]
    UNBIASED_POLICY = np.array([[1, 1, 1, 1, 1, 1, 1]] * Hyperparameters.AMOUNT_OF_GAMES_PER_WORKER)
    _oracleLoop(_predictWithUnbiasedValues, selfPlayPool)


def _predictWithUnbiasedValues(states):
    amountOfStates = len(states)
    return [UNBIASED_EVAL[:amountOfStates], UNBIASED_POLICY[:amountOfStates]]


# ***** Prediction with an optimized graph directly in tensorflow... *****
OPTIMIZED_GRAPH = None
INPUT_TENSOR = None
VALUE_OUT = None
POLICY_OUT = None


def _runOptimizedGraphOracle(model, selfPlayPool):
    global OPTIMIZED_GRAPH, VALUE_OUT, POLICY_OUT, INPUT_TENSOR, grad

    optiGraph, outputs = GraphOptimizer.createOptimizedGraph(model, K.get_session(), tf)
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1))) as sess:
            # read TensorRT model
            trt_graph = optiGraph

            # obtain the corresponding input-output tensor
            tf.import_graph_def(trt_graph, name='')
            
            a = model.layers[0]
            print(a)

            INPUT_TENSOR = sess.graph.get_tensor_by_name('InputLayer:0')
            POLICY_OUT = sess.graph.get_tensor_by_name('convexInputLayer:0')
            VALUE_OUT = sess.graph.get_tensor_by_name('ValueOut/Sigmoid:0')
            print(INPUT_TENSOR,POLICY_OUT,VALUE_OUT,grad)

            OPTIMIZED_GRAPH = sess
            _oracleLoop(_predictWithOptimizedGraph, selfPlayPool)


def _predictWithOptimizedGraph(states):

    b1 = 0.9
    b2 = 0.999
    lam = 0.5
    eps = 1e-8
    alpha = 0.01
    
    act = np.zeros((1,7))
    m = np.zeros_like(act)
    v = np.zeros_like(act)
    b1t, b2t = 1., 1.
    act_best, a_diff, f_best = [None]*3
    
    print(OPTIMIZED_GRAPH.run([VALUE_OUT], feed_dict={INPUT_TENSOR: np.array(states), POLICY_OUT:np.array(act)}))
    '''
    for i in range(50):
        f, g = OPTIMIZED_GRAPH.run([VALUE_OUT,grad], feed_dict={INPUT_TENSOR: np.array(states), POLICY_OUT:np.array(act)})
        if i == 0:
            act_best = act.copy()
            f_best = f.copy()
        else:
            prev_act_best = act_best.copy()
            I = (f < f_best)
            act_best[I] = act[I]
            f_best[I] = f[I]
            a_diff_i = np.mean(np.linalg.norm(act_best - prev_act_best, axis=1))
            a_diff = a_diff_i if a_diff is None else lam*a_diff + (1.-lam)*a_diff_i
                # print(a_diff_i, a_diff, np.sum(f))
            if a_diff < 1e-3 and i > 5:
                print('  + Adam took {} iterations'.format(i))
                f, g = OPTIMIZED_GRAPH.run([VALUE_OUT,grad], feed_dict={INPUT_TENSOR: np.array(states), POLICY_OUT:np.array(act_best)})
                return -f,act_best

        m = b1 * m + (1. - b1) * g
        v = b2 * v + (1. - b2) * (g * g)
        b1t *= b1
        b2t *= b2
        mhat = m/(1.-b1t)
        vhat = v/(1.-b2t)

        act -= alpha * mhat / (np.sqrt(v) + eps)
            # act = np.clip(act, -1, 1)
        act = np.clip(act, 0, 1)
        act = softmax(act)
        
    print('  + Warning: Adam did not converge.')
    f, g = OPTIMIZED_GRAPH.run([VALUE_OUT,grad], feed_dict={INPUT_TENSOR: np.array(states), POLICY_OUT:np.array(act)})
    return -f,act
    '''


def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x
