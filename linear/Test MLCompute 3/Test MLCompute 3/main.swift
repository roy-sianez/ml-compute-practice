//
//  main.swift
//  Test MLCompute 3
//
//  Created by Roy Sianez on 8/26/21.
//

import Foundation
import MLCompute

// Borrowed from MemoryToolkit
func copyData(
    from: UnsafeRawPointer,
    to: UnsafeMutableRawPointer,
    count: Int
) {
    to.copyMemory(from: from, byteCount: count)
}

typealias DataType = Float32
func arrayToTensorData(_ array: [DataType]) -> MLCTensorData {
    let byteCount = array.count * MemoryLayout<DataType>.stride
    let pointer = UnsafeMutableRawPointer.allocate(
        byteCount: byteCount,
        alignment: MemoryLayout<DataType>.alignment
    )
    copyData(from: array, to: pointer, count: byteCount)
    return .init(bytesNoCopy: pointer, length: byteCount)
}

let rawInData = Array(0 ..< 4)
let inData  = arrayToTensorData(rawInData
                                    .map { DataType($0) })
let outData = arrayToTensorData(rawInData
                                    .map { $0 * 2 + 5 }
                                    .map { DataType($0) })



let batchSize = rawInData.count

let device = MLCDevice.cpu()

let inputTensor = MLCTensor(
    descriptor: .init(
        shape: [batchSize, 1, 1, 1],
        dataType: .float32)!
)

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

let lossLabelsTensor = MLCTensor(
    descriptor: .init(
        shape: [batchSize, 1],
        dataType: .float32)!
)



let graph = MLCGraph()

let denseOutput = graph.node(
    with: MLCFullyConnectedLayer(
        weights: denseWeightsTensor,
        biases: denseBiasesTensor,
        descriptor: .init(
            kernelSizes: (1, 1),
            inputFeatureChannelCount: 1,
            outputFeatureChannelCount: 1))!,
    sources: [inputTensor]
)!



let trainingGraph = MLCTrainingGraph(
    graphObjects: [graph],
    lossLayer: MLCLossLayer(
        descriptor: .init(
            type: .meanSquaredError,
            reductionType: .mean)),
    optimizer: MLCAdamOptimizer(
        descriptor: .init(
            learningRate: 0.1,
            gradientRescale: 1,
            regularizationType: .none,
            regularizationScale: 0),
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-6,
        timeStep: 1
    )
)

trainingGraph.addInputs(
    ["in": inputTensor],
    lossLabels: ["lossLabels": lossLabelsTensor]
)

trainingGraph.compile(options: [], device: device)



for epoch in 0 ... 10_000 {
    trainingGraph.execute(
        inputsData: ["in": inData],
        lossLabelsData: ["lossLabels": outData],
        lossLabelWeightsData: nil,
        batchSize: batchSize,
        options: [.synchronous]
    ) { tensor, error, time in
        guard epoch % 1000 == 0 else { return }
        
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
        
        var result = [DataType]()
        for i in 0 ..< batchSize {
            result.append(
                bufferOutput
                    .advanced(by: i * dataTypeSize)
                    .load(as: DataType.self))
        }
        
        print(result)
    }
}
