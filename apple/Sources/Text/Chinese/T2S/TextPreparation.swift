import CoreML
import Foundation

public struct GPTSoVITST2SPreparedTextInput {
    public let sourceText: String
    public let normalizedText: String
    public let phoneIDs: [Int32]
    public let word2ph: [Int]
    public let phoneUnits: [GPTSoVITSTextPhoneUnit]
    public let bertTokens: [String]
    public let seq: MLMultiArray
    public let bert: MLMultiArray
    public let phoneCount: Int
    public let paddedPhoneCapacity: Int
    public let textFrontendBackend: String
}

public struct GPTSoVITST2SPreparedSegment {
    public let segment: GPTSoVITSTextSegment
    public let input: GPTSoVITST2SPreparedTextInput
}

public struct GPTSoVITST2SPreparedInputs {
    public let prompt: GPTSoVITST2SPreparedTextInput
    public let targets: [GPTSoVITST2SPreparedSegment]

    @available(*, deprecated, renamed: "prompt")
    public var reference: GPTSoVITST2SPreparedTextInput {
        prompt
    }

    public init(
        prompt: GPTSoVITST2SPreparedTextInput,
        targets: [GPTSoVITST2SPreparedSegment]
    ) {
        self.prompt = prompt
        self.targets = targets
    }

    @available(*, deprecated, message: "Use init(prompt:targets:).")
    public init(
        reference: GPTSoVITST2SPreparedTextInput,
        targets: [GPTSoVITST2SPreparedSegment]
    ) {
        self.init(prompt: reference, targets: targets)
    }
}

public enum GPTSoVITST2STextPreparationError: LocalizedError {
    case unsupportedLanguage(String)
    case missingWord2ph(String)
    case missingFixedShape(String)
    case invalidShape(String)
    case phoneCountExceedsCapacity(kind: String, phoneCount: Int, capacity: Int)

    public var errorDescription: String? {
        switch self {
        case let .unsupportedLanguage(language):
            return "GPTSoVITST2STextPreparer 暂不支持 language=\(language)。"
        case let .missingWord2ph(language):
            return "language=\(language) 的文本前端没有提供构造 BERT 所需的 word2ph。"
        case let .missingFixedShape(name):
            return "T2S 模型输入 \(name) 没有固定 multi-array shape。"
        case let .invalidShape(name):
            return "T2S 模型输入 \(name) 的 shape 不符合当前预期。"
        case let .phoneCountExceedsCapacity(kind, phoneCount, capacity):
            return "\(kind) phone_count=\(phoneCount) 超过 T2S 输入容量 \(capacity)。"
        }
    }
}

public final class GPTSoVITST2STextPreparer {
    public let t2s: T2SCoreMLDriver
    public let zhBert: ZhBertCharCoreMLDriver
    public let tokenizer: GPTSoVITSZhBertTokenizer
    public let frontend: GPTSoVITSTextFrontend
    public var g2pw: (any GPTSoVITSG2PWPredicting)? {
        g2pwStorage
    }
    public var textPhoneBackend: (any GPTSoVITSTextPhoneBackend)?

    private var g2pwStorage: (any GPTSoVITSG2PWPredicting)?
    private let g2pwFactory: (() throws -> (any GPTSoVITSG2PWPredicting)?)?

    public init(
        t2s: T2SCoreMLDriver,
        zhBert: ZhBertCharCoreMLDriver,
        tokenizer: GPTSoVITSZhBertTokenizer,
        frontend: GPTSoVITSTextFrontend = GPTSoVITSTextFrontend(),
        g2pw: (any GPTSoVITSG2PWPredicting)? = nil,
        g2pwFactory: (() throws -> (any GPTSoVITSG2PWPredicting)?)? = nil,
        textPhoneBackend: (any GPTSoVITSTextPhoneBackend)? = nil
    ) {
        self.t2s = t2s
        self.zhBert = zhBert
        self.tokenizer = tokenizer
        self.frontend = frontend
        self.g2pwStorage = g2pw
        self.g2pwFactory = g2pwFactory
        self.textPhoneBackend = textPhoneBackend
    }

    @available(*, deprecated, renamed: "preparePromptText(_:language:)")
    public func prepareReferenceText(
        _ text: String,
        language: GPTSoVITSTextLanguage = .zh
    ) throws -> GPTSoVITST2SPreparedTextInput {
        try preparePromptText(text, language: language)
    }

    public func preparePromptText(
        _ text: String,
        language: GPTSoVITSTextLanguage = .zh
    ) throws -> GPTSoVITST2SPreparedTextInput {
        let g2pwDriver = try resolvedG2PW(for: language)
        let phoneResult = try frontend.phoneResult(
            for: text,
            language: language,
            backend: textPhoneBackend,
            g2pwDriver: g2pwDriver
        )
        return try prepareSingleInput(
            sourceText: text,
            phoneResult: phoneResult,
            language: language,
            seqInputName: "ref_seq",
            bertInputName: "ref_bert",
            kind: "reference"
        )
    }

    public func prepareTargetSegments(
        text: String,
        language: GPTSoVITSTextLanguage = .zh,
        splitMethod: GPTSoVITSTextSplitMethod = .cut5
    ) throws -> [GPTSoVITST2SPreparedSegment] {
        let g2pwDriver = try resolvedG2PW(for: language)
        let prepared = try frontend.preprocessPhoneSegments(
            text: text,
            language: language,
            splitMethod: splitMethod,
            backend: textPhoneBackend,
            g2pwDriver: g2pwDriver
        )
        return try prepared.segmentResults.map { item in
            GPTSoVITST2SPreparedSegment(
                segment: item.segment,
                input: try prepareSingleInput(
                    sourceText: item.segment.text,
                    phoneResult: item.phoneResult,
                    language: language,
                    seqInputName: "text_seq",
                    bertInputName: "text_bert",
                    kind: "target"
                )
            )
        }
    }

    @available(*, deprecated, renamed: "preparePromptAndTargets(promptText:targetText:language:splitMethod:)")
    public func prepareReferenceAndTargets(
        referenceText: String,
        targetText: String,
        language: GPTSoVITSTextLanguage = .zh,
        splitMethod: GPTSoVITSTextSplitMethod = .cut5
    ) throws -> GPTSoVITST2SPreparedInputs {
        try preparePromptAndTargets(
            promptText: referenceText,
            targetText: targetText,
            language: language,
            splitMethod: splitMethod
        )
    }

    public func preparePromptAndTargets(
        promptText: String,
        targetText: String,
        language: GPTSoVITSTextLanguage = .zh,
        splitMethod: GPTSoVITSTextSplitMethod = .cut5
    ) throws -> GPTSoVITST2SPreparedInputs {
        GPTSoVITST2SPreparedInputs(
            prompt: try preparePromptText(promptText, language: language),
            targets: try prepareTargetSegments(text: targetText, language: language, splitMethod: splitMethod)
        )
    }

    private func prepareSingleInput(
        sourceText: String,
        phoneResult: GPTSoVITSTextPhoneResult,
        language: GPTSoVITSTextLanguage,
        seqInputName: String,
        bertInputName: String,
        kind: String
    ) throws -> GPTSoVITST2SPreparedTextInput {
        let seqShape = try fixedShape(for: seqInputName, in: t2s.prefillModel)
        let bertShape = try fixedShape(for: bertInputName, in: t2s.prefillModel)
        guard seqShape.count == 2, seqShape[0] == 1 else {
            throw GPTSoVITST2STextPreparationError.invalidShape(seqInputName)
        }
        guard bertShape.count == 2, bertShape[0] == 1024 else {
            throw GPTSoVITST2STextPreparationError.invalidShape(bertInputName)
        }
        let phoneCapacity = seqShape[1]
        guard bertShape[1] == phoneCapacity else {
            throw GPTSoVITST2STextPreparationError.invalidShape(bertInputName)
        }

        let phoneIDs = phoneResult.phoneIDs.map(Int32.init)
        let phoneCount = phoneIDs.count
        guard phoneCount <= phoneCapacity else {
            throw GPTSoVITST2STextPreparationError.phoneCountExceedsCapacity(
                kind: kind,
                phoneCount: phoneCount,
                capacity: phoneCapacity
            )
        }

        let paddedPhoneIDs = phoneIDs + Array(repeating: 0, count: phoneCapacity - phoneCount)
        let seq = try t2s.makeInt32Array(shape: seqShape, values: paddedPhoneIDs)

        let bertInput: GPTSoVITSZhBertCharInput?
        let bert: MLMultiArray
        if language.baseLanguage == "zh" {
            guard let word2ph = phoneResult.word2ph else {
                throw GPTSoVITST2STextPreparationError.missingWord2ph(language.rawValue)
            }
            let preparedBertInput = tokenizer.prepareInput(normalizedText: phoneResult.normalizedText)
            let featureResult = try zhBert.predictPhoneLevelFeature(input: preparedBertInput, word2ph: word2ph)
            bert = try copyPhoneLevelFeature(
                featureResult.phoneLevelFeature,
                targetShape: bertShape,
                activePhoneCount: phoneCount
            )
            bertInput = preparedBertInput
        } else {
            bert = try makeZeroBert(targetShape: bertShape)
            bertInput = nil
        }

        return GPTSoVITST2SPreparedTextInput(
            sourceText: sourceText,
            normalizedText: phoneResult.normalizedText,
            phoneIDs: phoneIDs,
            word2ph: phoneResult.word2ph ?? [],
            phoneUnits: phoneResult.phoneUnits,
            bertTokens: bertInput?.tokens ?? [],
            seq: seq,
            bert: bert,
            phoneCount: phoneCount,
            paddedPhoneCapacity: phoneCapacity,
            textFrontendBackend: phoneResult.backend
        )
    }

    private func resolvedG2PW(
        for language: GPTSoVITSTextLanguage
    ) throws -> (any GPTSoVITSG2PWPredicting)? {
        guard language.baseLanguage == "zh" else {
            return nil
        }
        if let g2pwStorage {
            return g2pwStorage
        }
        guard let g2pwFactory else {
            return nil
        }
        let driver = try g2pwFactory()
        g2pwStorage = driver
        return driver
    }

    private func fixedShape(for inputName: String, in model: MLModel) throws -> [Int] {
        guard let description = model.modelDescription.inputDescriptionsByName[inputName],
              let constraint = description.multiArrayConstraint else {
            throw GPTSoVITST2STextPreparationError.missingFixedShape(inputName)
        }
        let shape = constraint.shape.map { Int(truncating: $0) }
        guard !shape.isEmpty else {
            throw GPTSoVITST2STextPreparationError.missingFixedShape(inputName)
        }
        return shape
    }

    private func copyPhoneLevelFeature(
        _ source: MLMultiArray,
        targetShape: [Int],
        activePhoneCount: Int
    ) throws -> MLMultiArray {
        let sourceShape = source.shape.map { Int(truncating: $0) }
        guard sourceShape.count == 2, sourceShape[0] == targetShape[0] else {
            throw GPTSoVITST2STextPreparationError.invalidShape("phone_level_feature")
        }
        guard sourceShape[1] >= activePhoneCount, targetShape[1] >= activePhoneCount else {
            throw GPTSoVITST2STextPreparationError.invalidShape("phone_level_feature")
        }

        let target = try MLMultiArray(
            shape: targetShape.map(NSNumber.init(value:)),
            dataType: .float32
        )
        let targetBuffer = target.dataPointer.bindMemory(to: Float32.self, capacity: target.count)
        for index in 0..<target.count {
            targetBuffer[index] = 0
        }
        let hiddenSize = targetShape[0]
        for hiddenIndex in 0..<hiddenSize {
            for phoneIndex in 0..<activePhoneCount {
                let value = Self.value(in: source, indices: [hiddenIndex, phoneIndex])
                Self.setValue(value, in: target, indices: [hiddenIndex, phoneIndex])
            }
        }
        return target
    }

    private func makeZeroBert(targetShape: [Int]) throws -> MLMultiArray {
        let target = try MLMultiArray(
            shape: targetShape.map(NSNumber.init(value:)),
            dataType: .float32
        )
        let count = target.count
        let buffer = target.dataPointer.bindMemory(to: Float32.self, capacity: count)
        for index in 0..<count {
            buffer[index] = 0
        }
        return target
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
}
