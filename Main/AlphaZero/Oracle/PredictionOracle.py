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

    '''
    if (MachineSpecificSettings.IS_UNIX_MACHINE):
        _runOptimizedGraphOracle(model, selfPlayPool)
    else:
    '''
    _runNormalKerasOracle(model, selfPlayPool)

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


#NORMAL_MODEL = None


def _predictWithNormalModel(states):
    a = NORMAL_MODEL.predict([states])
    K.clear_session()
    return a


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


