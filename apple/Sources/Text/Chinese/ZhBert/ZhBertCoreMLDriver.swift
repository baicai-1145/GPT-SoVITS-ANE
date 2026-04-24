import CoreML
import Foundation

public struct ZhBertCharFeatureResult {
    public let charFeature: MLMultiArray
    public let phoneLevelFeature: MLMultiArray
}

public final class ZhBertCharCoreMLDriver {
    public enum OutputMode {
        case charFeature
        case phoneLevelFeature
    }

    public let modelURL: URL
    public let model: MLModel
    public let fixedTokenCapacity: Int?
    public let fixedPhoneCapacity: Int?
    public let outputMode: OutputMode

    public init(modelURL: URL, configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        self.modelURL = modelURL
        self.model = try GPTSoVITSRuntimeProfiler.measure("zhbert.init.model.load") {
            try Self.loadModel(at: modelURL, configuration: configuration)
        }
        self.fixedTokenCapacity = Self.profileFixedTokenCapacity(in: model)
        self.fixedPhoneCapacity = Self.profileFixedPhoneCapacity(in: model)
        self.outputMode = Self.profileOutputMode(in: model)
    }

    public func makeInt32Array(shape: [Int], values: [Int64]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map(NSNumber.init(value:)), dataType: .int32)
        guard array.count == values.count else {
            throw NSError(domain: "ZhBertCharCoreMLDriver", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Value count \(values.count) does not match shape element count \(array.count)."
            ])
        }
        for (index, value) in values.enumerated() {
            array[index] = NSNumber(value: Int32(truncatingIfNeeded: value))
        }
        return array
    }

    public func makeInputArrays(
        from input: GPTSoVITSZhBertCharInput
    ) throws -> (inputIDs: MLMultiArray, attentionMask: MLMultiArray, tokenTypeIDs: MLMultiArray) {
        let padded = try padInput(input)
        let shape = [1, padded.inputIDs.count]
        return (
            inputIDs: try makeInt32Array(shape: shape, values: padded.inputIDs),
            attentionMask: try makeInt32Array(shape: shape, values: padded.attentionMask),
            tokenTypeIDs: try makeInt32Array(shape: shape, values: padded.tokenTypeIDs)
        )
    }

    public func predictCharFeature(
        inputIDs: MLMultiArray,
        attentionMask: MLMultiArray,
        tokenTypeIDs: MLMultiArray
    ) throws -> MLMultiArray {
        guard outputMode == .charFeature else {
            throw NSError(domain: "ZhBertCharCoreMLDriver", code: 7, userInfo: [
                NSLocalizedDescriptionKey: "Current model emits phone_level_feature directly and does not expose char_feature."
            ])
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIDs),
            "attention_mask": MLFeatureValue(multiArray: attentionMask),
            "token_type_ids": MLFeatureValue(multiArray: tokenTypeIDs),
        ])
        let output = try model.prediction(from: provider)
        guard let value = output.featureValue(for: "char_feature")?.multiArrayValue else {
            throw NSError(domain: "ZhBertCharCoreMLDriver", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Missing MLMultiArray output 'char_feature'."
            ])
        }
        return value
    }

    public func predictCharFeature(input: GPTSoVITSZhBertCharInput) throws -> MLMultiArray {
        let arrays = try makeInputArrays(from: input)
        return try predictCharFeature(
            inputIDs: arrays.inputIDs,
            attentionMask: arrays.attentionMask,
            tokenTypeIDs: arrays.tokenTypeIDs
        )
    }

    public func buildPhoneLevelFeature(
        charFeature: MLMultiArray,
        word2ph: [Int]
    ) throws -> MLMultiArray {
        let shape = charFeature.shape.map { Int(truncating: $0) }
        guard shape.count == 2 else {
            throw NSError(domain: "ZhBertCharCoreMLDriver", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "Expected char_feature rank 2, got shape \(shape)."
            ])
        }

        let availableCharCount = shape[0]
        let hiddenSize = shape[1]
        let charCount = word2ph.count
        guard availableCharCount >= charCount else {
            throw NSError(domain: "ZhBertCharCoreMLDriver", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "word2ph length \(charCount) exceeds char_feature char count \(availableCharCount)."
            ])
        }

        let phoneCount = word2ph.reduce(0, +)
        let phoneLevelFeature = try MLMultiArray(
            shape: [NSNumber(value: hiddenSize), NSNumber(value: phoneCount)],
            dataType: .float32
        )

        var phoneCursor = 0
        for charIndex in 0..<charCount {
            let repeatCount = word2ph[charIndex]
            guard repeatCount >= 0 else {
                throw NSError(domain: "ZhBertCharCoreMLDriver", code: 5, userInfo: [
                    NSLocalizedDescriptionKey: "Negative repeat count at char index \(charIndex): \(repeatCount)."
                ])
            }
            for _ in 0..<repeatCount {
                for hiddenIndex in 0..<hiddenSize {
                    let value = Self.value(in: charFeature, indices: [charIndex, hiddenIndex])
                    Self.setValue(value, in: phoneLevelFeature, indices: [hiddenIndex, phoneCursor])
                }
                phoneCursor += 1
            }
        }
        return phoneLevelFeature
    }

    public func predictPhoneLevelFeature(
        input: GPTSoVITSZhBertCharInput,
        word2ph: [Int]
    ) throws -> ZhBertCharFeatureResult {
        switch outputMode {
        case .charFeature:
            let charFeature = try predictCharFeature(input: input)
            let phoneLevelFeature = try buildPhoneLevelFeature(charFeature: charFeature, word2ph: word2ph)
            return ZhBertCharFeatureResult(
                charFeature: charFeature,
                phoneLevelFeature: phoneLevelFeature
            )
        case .phoneLevelFeature:
            let arrays = try makeInputArrays(from: input, word2ph: word2ph)
            let phoneLevelFeature = try predictDirectPhoneLevelFeature(
                inputIDs: arrays.inputIDs,
                attentionMask: arrays.attentionMask,
                tokenTypeIDs: arrays.tokenTypeIDs,
                word2ph: arrays.word2ph
            )
            return ZhBertCharFeatureResult(
                charFeature: phoneLevelFeature,
                phoneLevelFeature: phoneLevelFeature
            )
        }
    }

    public func predictPhoneLevelFeature(
        phoneResult: GPTSoVITSChinesePhoneResult,
        tokenizer: GPTSoVITSZhBertTokenizer
    ) throws -> ZhBertCharFeatureResult {
        try predictPhoneLevelFeature(
            input: tokenizer.prepareInput(phoneResult: phoneResult),
            word2ph: phoneResult.word2ph
        )
    }

    private static func loadModel(at url: URL, configuration: MLModelConfiguration) throws -> MLModel {
        try GPTSoVITSCoreMLModelLoader.loadModel(at: url, configuration: configuration)
    }

    private static func profileFixedTokenCapacity(in model: MLModel) -> Int? {
        GPTSoVITSRuntimeProfiler.measure("zhbert.init.fixed_token_capacity.resolve") {
            Self.resolveFixedTokenCapacity(in: model)
        }
    }

    private static func profileFixedPhoneCapacity(in model: MLModel) -> Int? {
        GPTSoVITSRuntimeProfiler.measure("zhbert.init.fixed_phone_capacity.resolve") {
            Self.resolveFixedPhoneCapacity(in: model)
        }
    }

    private static func profileOutputMode(in model: MLModel) -> OutputMode {
        GPTSoVITSRuntimeProfiler.measure("zhbert.init.output_mode.resolve") {
            Self.resolveOutputMode(in: model)
        }
    }

    private func makeInputArrays(
        from input: GPTSoVITSZhBertCharInput,
        word2ph: [Int]
    ) throws -> (inputIDs: MLMultiArray, attentionMask: MLMultiArray, tokenTypeIDs: MLMultiArray, word2ph: MLMultiArray) {
        let tokenArrays = try makeInputArrays(from: input)
        let paddedWord2ph = try padWord2ph(word2ph)
        return (
            inputIDs: tokenArrays.inputIDs,
            attentionMask: tokenArrays.attentionMask,
            tokenTypeIDs: tokenArrays.tokenTypeIDs,
            word2ph: try makeInt32Array(shape: [paddedWord2ph.count], values: paddedWord2ph.map(Int64.init))
        )
    }

    private func predictDirectPhoneLevelFeature(
        inputIDs: MLMultiArray,
        attentionMask: MLMultiArray,
        tokenTypeIDs: MLMultiArray,
        word2ph: MLMultiArray
    ) throws -> MLMultiArray {
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIDs),
            "attention_mask": MLFeatureValue(multiArray: attentionMask),
            "token_type_ids": MLFeatureValue(multiArray: tokenTypeIDs),
            "word2ph": MLFeatureValue(multiArray: word2ph),
        ])
        let output = try model.prediction(from: provider)
        guard let value = output.featureValue(for: "phone_level_feature")?.multiArrayValue else {
            throw NSError(domain: "ZhBertCharCoreMLDriver", code: 8, userInfo: [
                NSLocalizedDescriptionKey: "Missing MLMultiArray output 'phone_level_feature'."
            ])
        }
        return value
    }

    private func padInput(_ input: GPTSoVITSZhBertCharInput) throws -> GPTSoVITSZhBertCharInput {
        guard let fixedTokenCapacity else {
            return input
        }
        guard input.tokenCount <= fixedTokenCapacity else {
            throw NSError(domain: "ZhBertCharCoreMLDriver", code: 6, userInfo: [
                NSLocalizedDescriptionKey: "Token count \(input.tokenCount) exceeds current zh_bert Core ML export capacity \(fixedTokenCapacity)."
            ])
        }
        if input.tokenCount == fixedTokenCapacity {
            return input
        }

        let paddingCount = fixedTokenCapacity - input.tokenCount
        return GPTSoVITSZhBertCharInput(
            normalizedText: input.normalizedText,
            tokens: input.tokens + Array(repeating: "[PAD]", count: paddingCount),
            inputIDs: input.inputIDs + Array(repeating: input.padTokenID, count: paddingCount),
            attentionMask: input.attentionMask + Array(repeating: 0, count: paddingCount),
            tokenTypeIDs: input.tokenTypeIDs + Array(repeating: 0, count: paddingCount),
            padTokenID: input.padTokenID
        )
    }

    private func padWord2ph(_ word2ph: [Int]) throws -> [Int] {
        guard let fixedPhoneCapacity else {
            return word2ph
        }
        let phoneCount = word2ph.reduce(0, +)
        guard phoneCount <= fixedPhoneCapacity else {
            throw NSError(domain: "ZhBertCharCoreMLDriver", code: 9, userInfo: [
                NSLocalizedDescriptionKey: "Phone count \(phoneCount) exceeds current zh_bert Core ML export capacity \(fixedPhoneCapacity)."
            ])
        }
        guard let fixedTokenCapacity else {
            return word2ph
        }
        let charCapacity = max(fixedTokenCapacity - 2, 0)
        guard word2ph.count <= charCapacity else {
            throw NSError(domain: "ZhBertCharCoreMLDriver", code: 10, userInfo: [
                NSLocalizedDescriptionKey: "word2ph length \(word2ph.count) exceeds current zh_bert Core ML export char capacity \(charCapacity)."
            ])
        }
        return word2ph + Array(repeating: 0, count: charCapacity - word2ph.count)
    }

    private static func value(in array: MLMultiArray, indices: [Int]) -> Float {
        let offset = linearOffset(for: indices, in: array)
        switch array.dataType {
        case .float32:
            return array.dataPointer.bindMemory(to: Float32.self, capacity: offset + 1)[offset]
        case .double:
            return Float(array.dataPointer.bindMemory(to: Double.self, capacity: offset + 1)[offset])
        case .float16:
            return Float(array[offset].floatValue)
        default:
            return array[offset].floatValue
        }
    }

    private static func setValue(_ value: Float, in array: MLMultiArray, indices: [Int]) {
        let offset = linearOffset(for: indices, in: array)
        switch array.dataType {
        case .float32:
            array.dataPointer.bindMemory(to: Float32.self, capacity: offset + 1)[offset] = value
        case .double:
            array.dataPointer.bindMemory(to: Double.self, capacity: offset + 1)[offset] = Double(value)
        default:
            array[offset] = NSNumber(value: value)
        }
    }

    private static func linearOffset(for indices: [Int], in array: MLMultiArray) -> Int {
        precondition(indices.count == array.shape.count, "Rank mismatch when indexing MLMultiArray.")
        var offset = 0
        for (index, stride) in zip(indices, array.strides) {
            offset += index * Int(truncating: stride)
        }
        return offset
    }

    private static func resolveFixedTokenCapacity(in model: MLModel) -> Int? {
        if usesDynamicTokenInput(in: model) {
            return nil
        }
        guard let description = model.modelDescription.inputDescriptionsByName["input_ids"],
              let constraint = description.multiArrayConstraint else {
            return nil
        }
        let shape = constraint.shape.map { Int(truncating: $0) }
        guard shape.count == 2 else {
            return nil
        }
        return shape[1]
    }

    private static func usesDynamicTokenInput(in model: MLModel) -> Bool {
        guard let creatorDefined = model.modelDescription.metadata[.creatorDefinedKey] as? [String: Any] else {
            return false
        }
        guard let rawValue = creatorDefined["gpt_sovits_dynamic_token_dim"] as? String else {
            return false
        }
        return rawValue.lowercased() == "true"
    }

    private static func resolveFixedPhoneCapacity(in model: MLModel) -> Int? {
        guard let description = model.modelDescription.outputDescriptionsByName["phone_level_feature"],
              let constraint = description.multiArrayConstraint else {
            return nil
        }
        let shape = constraint.shape.map { Int(truncating: $0) }
        guard shape.count == 2 else {
            return nil
        }
        return shape[1]
    }

    private static func resolveOutputMode(in model: MLModel) -> OutputMode {
        if model.modelDescription.outputDescriptionsByName["phone_level_feature"] != nil {
            return .phoneLevelFeature
        }
        return .charFeature
    }
}
