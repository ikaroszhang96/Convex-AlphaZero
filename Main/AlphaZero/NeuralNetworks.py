from Main import Hyperparameters, MachineSpecificSettings
import keras


def _createBatchNormalizedConvBlock(inputNode, filters):
    temp = keras.layers.Conv2D(filters, kernel_size=(4, 4), strides=(1, 1), padding='SAME')(inputNode)
    temp = keras.layers.BatchNormalization()(temp)
    return keras.layers.ReLU()(temp)


def createResidualNetwork(inputShape, filtersPerConv, convPerResidual, amountOfResidualBlocks):
    inputLayer = keras.layers.Input(inputShape, name="InputLayer")
    policy = keras.layers.Input((7,), name="convex")
    l2Reg = Hyperparameters.L2_REGULARIZATION
    Conv2D = keras.layers.Conv2D
    
    us = []
    zs = []
    z_zs = []
    z_ys = []
    z_us = []

    # Build Residual Tower
    skipConnection = _createBatchNormalizedConvBlock(inputLayer, filtersPerConv)
    x = skipConnection
    y = policy
    for i in range(amountOfResidualBlocks - 1):
        resBlock = skipConnection
        for j in range(convPerResidual):
            resBlock = Conv2D(filtersPerConv, kernel_size=(4, 4), strides=(1, 1), padding='SAME')(
                resBlock)
            resBlock = keras.layers.BatchNormalization()(resBlock)

            # If it's the last conv Layer in the residual block we wait with adding the ReLU until we have
            # added the skip connection
            if (j + 1 < convPerResidual):
                resBlock = keras.layers.ReLU()(resBlock)

        skipConnection = keras.layers.add([skipConnection, resBlock])
        skipConnection = keras.layers.ReLU()(skipConnection)
        us.append(skipConnection)
        
        
    prevU, prevZ = x, y
    for i in range(amountOfResidualBlocks):
        z_add = []
        if i > 0:
            zu_u = keras.layers.Flatten()(prevU)
            zu_u = keras.layers.Dense(7, activation='relu',use_bias=True,
                              kernel_regularizer=regularizers.l2(0.01),
                              bias_regularizer=regularizers.l2(0.01), bias_init='Ones')(zu_u)
            
            if i == (amountOfResidualBlocks-1):
                z_zu = keras.layers.Dense(1,use_bias=False, kernel_regularizer=regularizers.l2(0.01))(keras.layers.Multiply([zu_u, prevZ]))
            else:
                z_zu = keras.layers.Dense(7,use_bias=False, kernel_regularizer=regularizers.l2(0.01))(keras.layers.Multiply([zu_u, prevZ]))
            z_zs.append(z_zu)
            z_add.append(z_zu)
        
        yu_u = keras.layers.Flatten()(prevU)
        yu_u = keras.layers.Dense(7, use_bias=True, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), bias_init='Ones')(yu_u)
        if i == (amountOfResidualBlocks-1):
            z_yu = keras.layers.Dense(1, use_bias=False, kernel_regularizer=regularizers.l2(0.01))(keras.layers.Multiply([yu_u, y]))
        else:
            z_yu = keras.layers.Dense(7, use_bias=False, kernel_regularizer=regularizers.l2(0.01))(keras.layers.Multiply([yu_u, y]))
        z_ys.append(z_yu)
        z_add.append(z_yu)
        
        z_u = keras.layers.Flatten()(prevU)
        if i == (amountOfResidualBlocks-1):
            z_u = keras.layers.Dense(1, use_bias=True, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), bias_init='Zeros')(z_u)
        else:
            z_u = keras.layers.Dense(7, use_bias=True, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), bias_init='Zeros')(z_u)
        z_us.append(z_u)
        z_add.append(z_u)
        
        z = keras.layers.add(z_add)
        
        if i < (amountOfResidualBlocks - 1):
            z = keras.layers.LeakyReLU(alpha=0.1)(z)
        else:
            z = keras.activations.sigmoid(z)
            
        zs.append(z)
        prevU = us[i] if i < (amountOfResidualBlocks - 1) else None
        prevZ = z

    # Evaluation Head
    evalHead = z


    # Create Full Model
    model = keras.Model(inputs = [inputLayer, policy], evalHead)
    _compileRezNetModel(model)
    print("Created Rez-Net model")

    return model


def createMultipleGPUModel(model):
    gpuModel = keras.utils.multi_gpu_model(model, gpus=MachineSpecificSettings.AMOUNT_OF_GPUS)
    _compileRezNetModel(gpuModel)
    print("Created Multiple GPU Rez-Net model, using {} GPU's".format(MachineSpecificSettings.AMOUNT_OF_GPUS))
    return gpuModel


def _compileRezNetModel(model):
    # optimizer = keras.optimizers.Adam()
    optimizer = keras.optimizers.SGD(lr=Hyperparameters.LEARNING_RATE, momentum=Hyperparameters.MOMENTUM)
    model.compile(optimizer, ['mse', 'categorical_crossentropy'])
