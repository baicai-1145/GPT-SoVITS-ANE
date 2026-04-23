import CoreML
import Foundation

public final class CNHubertCoreMLDriver {
    public let model: MLModel
    public let fixedInputSampleCount: Int?
    public let fixedOutputShape: [Int]?

    public init(modelURL: URL, configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        self.model = try Self.loadModel(at: modelURL, configuration: configuration)
        self.fixedInputSampleCount = Self.resolveFixedTrailingDimension(inputName: "input_values", in: model)
        self.fixedOutputShape = Self.resolveFixedShape(outputName: "ssl_content", in: model)
    }

    public func makeFloat32Array(shape: [Int], values: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map(NSNumber.init(value:)), dataType: .float32)
        guard array.count == values.count else {
            throw NSError(domain: "CNHubertCoreMLDriver", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Value count \(values.count) does not match shape element count \(array.count)."
            ])
        }
        for (index, value) in values.enumerated() {
            array[index] = NSNumber(value: value)
        }
        return array
    }

    public func predictSSLContent(inputValues: MLMultiArray) throws -> MLMultiArray {
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_values": MLFeatureValue(multiArray: inputValues),
        ])
        let output = try model.prediction(from: provider)
        guard let value = output.featureValue(for: "ssl_content")?.multiArrayValue else {
            throw NSError(domain: "CNHubertCoreMLDriver", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Missing MLMultiArray output 'ssl_content'."
            ])
        }
        return value
    }

    private static func loadModel(at url: URL, configuration: MLModelConfiguration) throws -> MLModel {
        try GPTSoVITSCoreMLModelLoader.loadModel(at: url, configuration: configuration)
    }

    private static func resolveFixedTrailingDimension(inputName: String, in model: MLModel) -> Int? {
        guard let shape = resolveFixedShape(inputName: inputName, in: model), shape.count >= 1 else {
            return nil
        }
        return shape.last
    }

    private static func resolveFixedShape(inputName: String, in model: MLModel) -> [Int]? {
        guard let description = model.modelDescription.inputDescriptionsByName[inputName],
              let constraint = description.multiArrayConstraint else {
            return nil
        }
        let shape = constraint.shape.map { Int(truncating: $0) }
        return shape.isEmpty ? nil : shape
    }

    private static func resolveFixedShape(outputName: String, in model: MLModel) -> [Int]? {
        guard let description = model.modelDescription.outputDescriptionsByName[outputName],
              let constraint = description.multiArrayConstraint else {
            return nil
        }
        let shape = constraint.shape.map { Int(truncating: $0) }
        return shape.isEmpty ? nil : shape
    }
}
