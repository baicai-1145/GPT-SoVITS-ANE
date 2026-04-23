import CoreML
import Foundation

public final class SpeakerEncoderCoreMLDriver {
    public let model: MLModel

    public init(modelURL: URL, configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        self.model = try GPTSoVITSRuntimeProfiler.measure("speaker_encoder.init.model.load") {
            try Self.loadModel(at: modelURL, configuration: configuration)
        }
    }

    public func makeZeroFloat32Array(shape: [Int]) throws -> MLMultiArray {
        try MLMultiArray(shape: shape.map(NSNumber.init(value:)), dataType: .float32)
    }

    public func embed(fbank80: MLMultiArray) throws -> MLMultiArray {
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "fbank_80": MLFeatureValue(multiArray: fbank80),
        ])
        let output = try model.prediction(from: provider)
        guard let value = output.featureValue(for: "sv_emb")?.multiArrayValue else {
            throw NSError(domain: "SpeakerEncoderCoreMLDriver", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Missing MLMultiArray output 'sv_emb'."
            ])
        }
        return value
    }

    private static func loadModel(at url: URL, configuration: MLModelConfiguration) throws -> MLModel {
        try GPTSoVITSCoreMLModelLoader.loadModel(at: url, configuration: configuration)
    }
}
