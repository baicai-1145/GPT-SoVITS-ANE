import CoreML
import Foundation

public final class SSLLatentExtractorCoreMLDriver {
    public let model: MLModel
    public let fixedInputShape: [Int]?
    public let fixedPromptCount: Int?

    public init(modelURL: URL, configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        self.model = try Self.loadModel(at: modelURL, configuration: configuration)
        self.fixedInputShape = Self.resolveFixedShape(inputName: "ssl_content", in: model)
        self.fixedPromptCount = Self.resolveFixedTrailingDimension(outputName: "prompt_semantic", in: model)
    }

    public func predictPromptSemantic(sslContent: MLMultiArray) throws -> MLMultiArray {
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "ssl_content": MLFeatureValue(multiArray: sslContent),
        ])
        let output = try model.prediction(from: provider)
        guard let value = output.featureValue(for: "prompt_semantic")?.multiArrayValue else {
            throw NSError(domain: "SSLLatentExtractorCoreMLDriver", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Missing MLMultiArray output 'prompt_semantic'."
            ])
        }
        return value
    }

    public func promptSemanticValues(sslContent: MLMultiArray) throws -> [Int32] {
        let array = try predictPromptSemantic(sslContent: sslContent)
        return (0..<array.count).map { Int32(truncating: array[$0]) }
    }

    private static func loadModel(at url: URL, configuration: MLModelConfiguration) throws -> MLModel {
        try GPTSoVITSCoreMLModelLoader.loadModel(at: url, configuration: configuration)
    }

    private static func resolveFixedTrailingDimension(outputName: String, in model: MLModel) -> Int? {
        guard let shape = resolveFixedShape(outputName: outputName, in: model), shape.count >= 1 else {
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
