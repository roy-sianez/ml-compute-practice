//
//  main.swift
//  Test MLCompute 3
//
//  Created by Roy Sianez on 8/26/21.
//

///
/// This is my personal favorite "hello, world" project for machine
/// learning. The goal is to perform linear regression over a simple
/// dataset. In this case, we have
///
///     in  = [0, 1, 2, 3]
///     out = [5, 7, 9, 11]
///
/// The model that we are training has only one dense layer with
/// one weight and one bias. If everything goes correctly, the weight
/// should be 2 and the bias should be 5. Note that
///
///     out == in.map { ($0 * 2) + 5 }
///
/// The constants in that code correspond to the weight and bias.
/// A linear regression model like this one can be represented by the
/// following linear equation:
///
///     out = (in * weight) + bias
///


import Foundation
import MLCompute


// MARK: Utility functions

typealias DataType = Float32

/// Convert an array of values to an instance of `MLCTensorData`.
func arrayToTensorData(_ array: [DataType]) -> MLCTensorData {
    let byteCount = array.count * MemoryLayout<DataType>.stride
    let pointer = UnsafeMutableRawPointer.allocate(
        byteCount: byteCount,
        alignment: MemoryLayout<DataType>.alignment
    )
    array.withUnsafeBytes { bufferPointer in
        pointer.copyMemory(
            from: bufferPointer.baseAddress!,
            byteCount: byteCount)
    }
    return .init(bytesNoCopy: pointer, length: byteCount)
}


// MARK: Define training data

/// To keep things simple, we have only 4 training examples.
let rawInData = Array(0 ..< 4)
let inData  = arrayToTensorData(rawInData
                                    .map { DataType($0) })
let outData = arrayToTensorData(rawInData
                                    .map { $0 * 2 + 5 }
                                    .map { DataType($0) })


// MARK: Define model graph

/// To keep things simple, we train on every example for a batch.
let batchSize = rawInData.count

// For the input tensor, be sure to include the batch size as the first dimension of the tensor. The shape of the input tensor matches the shape of the data that the framework passes as input to the model for each BATCH.
let inputTensor = MLCTensor(
    descriptor: .init(
        shape: [batchSize, 1, 1, 1],
        dataType: .float32)!
)

// These are the weight and bias tensors for the model's dense layer. The weights and biases are NOT related to the batch size; they are applied to each element individually. Therefore, the weights and biases do NOT include any reference to `batchSize` in the definitions of their shapes.
// There are separate initializers to conveniently create the tensor descriptors for weights and biases for different types of layers; here, for simplicity, we use array literals and the init(shape:dataType:) initializer instead.
// The weights and biases for the one dense layer in the model are actually single numbers, because the model is a linear equation (y = wx + b). Correspondingly, every dimension in the shape is 1. Nonetheless, 4 dimensions are required for the weights and biases of a dense layer.
// We use the `fillWithData` initializer here so that, when the model is trained, these tensors will initially be filled with 0s (since they represent scalars, in this case, their values will be 0). There is also a separate `MLCTensor` initializer that allows you to ask for the tensor to be filled with random data, which is necessary for training anything more complicated than a linear regression model.
let denseWeightsTensor = MLCTensor(
    descriptor: .init(
        shape: [1, 1, 1, 1],
        dataType: .float32)!,
    fillWithData: 0
)
let denseBiasesTensor = MLCTensor(
    descriptor: .init(
        shape: [1, 1, 1, 1],
        dataType: .float32)!,
    fillWithData: 0
)

// The loss labels are the "correct output" for the model. This tensor describes the shape of the loss labels that will be provided to the model at execution time. The first dimension of this shape MUST be the batch size, and there MUST be at least one more dimension describing the shape of each individual loss label corresponding to an individual example. In many common cases, the shape will have two dimensions. In this case, the second dimension is 1, because the desired output for 1 example is in the form of a scalar.
let lossLabelsTensor = MLCTensor(
    descriptor: .init(
        shape: [batchSize, 1],
        dataType: .float32)!
)

/// The graph object representing the model.
let graph = MLCGraph()

// Add a dense layer to the model. Create the `MLCFullyConnectedLayer` and pass into it tensors representing the shapes of its weights and biases. Pass `inputTensor` as a source, because this is the input data to the dense layer. The `graph.node` function returns a tensor representing the output of the dense layer. (A fully connected layer is the same thing as a dense layer.)
// The descriptor describes the dense layer — in this case, the input and output dimensions are both 1.
// For a dense layer, it is my best guess that the `width` and `height` value for the `kernelSizes` parameter of the descriptor should correspond to the values for the `inputFeatureChannelCount` and `outputFeatureChannelCount` parameters. (For this example, setting this to `(1, 1)` seems to work.) It is my best guess that this parameter could be changed for a layer that performs convolution on image data to customize the way that convolution is done.
let denseOutput = graph.node(
    with: MLCFullyConnectedLayer(
        weights: denseWeightsTensor,
        biases: denseBiasesTensor,
        descriptor: .init(
            kernelSizes: (1, 1),
            inputFeatureChannelCount: 1,
            outputFeatureChannelCount: 1))!,
    source: inputTensor
)!


// MARK: Train model

// Define the device used for training
let device = MLCDevice.gpu()!
// Comment the previous line and uncomment the following line to use the CPU instead:
// let device = MLCDevice.cpu()

// Create the training graph
// Pass `graph` to the initializer to use the model graph that we have created for training.
// The `lossLayer` parameter defines how the framework measures loss while training the model.
// The `optimizer` parameter defines which optimizer is used for training. We customize hyperparameters such as learning rate, gradient scaling, and regularization.
let trainingGraph = MLCTrainingGraph(
    graphObjects: [graph],
    lossLayer: MLCLossLayer(
        descriptor: .init(
            type: .meanSquaredError,
            reductionType: .mean)),
    optimizer: MLCAdamOptimizer(
        descriptor: .init(
            learningRate: 0.3,
            gradientRescale: 1,
            regularizationType: .none,
            regularizationScale: 0))
)

// Add inputs and loss labels to the training graph.
// The inputs are the tensors describing the shape of the data that we will pass to the training graph to use as INPUTS to our model.
// The loss labels are the tensors describing the shape of the data that we will pass to the training graph to use as DESIRED OUTPUTS to our model.
// Use the same string key for the tensors describing this data, and for the corresponding actual data that we will pass in `trainingGraph.execute`.
trainingGraph.addInputs(
    ["in": inputTensor],
    lossLabels: ["lossLabels": lossLabelsTensor]
)

// "Compile" the graph. Theoretically, it is here that the framework checks the graph for validity, but I have found that MLCompute crashes with an obscure error as a result of an invalid graph MUCH more often than it emits a useful error at this stage.
trainingGraph.compile(options: [], device: device)

// Train the model for 500 epochs
for epoch in 0 ... 500 {
    // This function trains the model for one iteration.
    // Pass our `inData` tensor data for INPUT data.
    // Pass our `outData` tensor for DESIRED OUTPUT data.
    // Note that we use the same corresponding string keys as above.
    // Pass `nil` for losslabelWeightsData (it is my best guess that this is not needed unless you want to have the framework make bigger optimizer updates for some training examples than for others. This would be useful functionality if, for example, you were building a classifier model and had a lot more examples for one class than another. Then you would have the framework make smaller updates for examples which belonged to the class with extra examples.)
    // Pass the correct batch size.
    // Pass `.synchronous` as an option if you want the function to wait to return until the training iteration has completed.
    trainingGraph.execute(
        inputsData: ["in": inData],
        lossLabelsData: ["lossLabels": outData],
        lossLabelWeightsData: nil,
        batchSize: batchSize,
        options: [.synchronous]
    ) { tensor, error, time in
        // Inside the closure, which is called when an iteration completes, we print debugging data every 100 iterations.
        // We don't actually use the closure parameters in this case, but I've included them for illustrative purposes. `tensor` is a curious parameter — I don't know what it does.
        // We continue to print debugging data every 100 iterations; otherwise, we return:
        guard epoch % 100 == 0 else { return }
        
        // Retrieve the outputs of the dense layer for this batch. The output shape is naturally the same as the shape for our `lossLabelsTensor`: [batchSize, 1]
        let dataTypeSize = MemoryLayout<DataType>.stride
        let numBytes = batchSize * dataTypeSize
        let bufferOutput = UnsafeMutableRawPointer.allocate(
            byteCount: numBytes,
            alignment: MemoryLayout<DataType>.alignment
        )
        denseOutput.copyDataFromDeviceMemory(
            toBytes: bufferOutput,
            length: numBytes,
            synchronizeWithDevice: false
        )
        
        // Convert the data referenced by the pointer to a Swift array and print it. What is printed is the predicted output data for this specific iteration.
        var result = [DataType]()
        for i in 0 ..< batchSize {
            result.append(
                bufferOutput
                    .advanced(by: i * dataTypeSize)
                    .load(as: DataType.self))
        }
        
        print(result)
        
        bufferOutput.deallocate()
    }
}


// MARK: Get trained weights and biases

// Only needed for GPU
// This takes the weights and biases for all the trained layers from GPU memory and copies them to CPU memory. (I am creating this example on an M1 Mac and for some reason, it is still necessary to call this function when training on GPU, even though M1 Macs have a unified memory archetecture.)
trainingGraph.synchronizeUpdates()

// Retrieve and print the trained weights and biases. If you were training a machine learning model for practical purposes, you would probably save the weights and biases for all your layers to disk at this point.
// This should print that the weight is 2 and the bias is 5. If it does, the model has trained correctly!
denseWeightsTensor.data!.withUnsafeBytes { buffer in
    print("Weight: \(buffer.baseAddress!.load(as: DataType.self))")
}
denseBiasesTensor.data!.withUnsafeBytes { buffer in
    print("Bias: \(buffer.baseAddress!.load(as: DataType.self))")
}
