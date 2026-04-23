import CoreML
import Foundation

public struct GPTSoVITSSpeakerConditionedResult {
    public let semanticCodes: [Int32]
    public let t2sState: T2SDecodeState
    public let vitsResult: VITSDecodeResult
}

private struct GPTSoVITSResolvedSemanticCodes {
    let semanticCodes: [Int32]
    let paddedCodes: [Int32]
    let validCount: Int
}

private struct GPTSoVITSResolvedVITSText {
    let array: MLMultiArray
    let phoneCount: Int
}

public enum GPTSoVITSSpeakerConditionedPipelineError: LocalizedError {
    case missingFixedShape(String)
    case invalidShape(String)
    case vitsTextPhoneCountExceedsCapacity(phoneCount: Int, capacity: Int)

    public var errorDescription: String? {
        switch self {
        case let .missingFixedShape(name):
            return "VITS 模型输入 \(name) 没有固定 multi-array shape。"
        case let .invalidShape(name):
            return "VITS 模型输入 \(name) 的 shape 不符合当前预期。"
        case let .vitsTextPhoneCountExceedsCapacity(phoneCount, capacity):
            return "VITS text phone_count=\(phoneCount) 超过输入容量 \(capacity)。"
        }
    }
}

public final class GPTSoVITSSpeakerConditionedPipeline {
    public let t2s: T2SCoreMLDriver
    public let vits: VITSSpeakerConditioningDriver

    public init(
        t2sBundleDirectory: URL,
        vitsBundleDirectory: URL,
        configuration: MLModelConfiguration = MLModelConfiguration()
    ) throws {
        self.t2s = try GPTSoVITSRuntimeProfiler.measure("speaker_conditioned.t2s.init") {
            try T2SCoreMLDriver(
                bundleDirectory: t2sBundleDirectory,
                configuration: configuration
            )
        }
        self.vits = try GPTSoVITSRuntimeProfiler.measure("speaker_conditioned.vits.init") {
            try VITSSpeakerConditioningDriver(
                vitsBundleDirectory: vitsBundleDirectory,
                configuration: configuration
            )
        }
    }

    public init(
        t2sBundleDirectory: URL,
        speakerEncoderModelURL: URL,
        vitsBundleDirectory: URL,
        configuration: MLModelConfiguration = MLModelConfiguration()
    ) throws {
        self.t2s = try GPTSoVITSRuntimeProfiler.measure("speaker_conditioned.t2s.init") {
            try T2SCoreMLDriver(
                bundleDirectory: t2sBundleDirectory,
                configuration: configuration
            )
        }
        self.vits = try GPTSoVITSRuntimeProfiler.measure("speaker_conditioned.vits.init") {
            try VITSSpeakerConditioningDriver(
                speakerEncoderModelURL: speakerEncoderModelURL,
                vitsBundleDirectory: vitsBundleDirectory,
                configuration: configuration
            )
        }
    }

    public func synthesize(
        prompts: MLMultiArray,
        promptLength: Int? = nil,
        refSeq: MLMultiArray,
        refSeqLength: Int? = nil,
        textSeq: MLMultiArray,
        textSeqLength: Int? = nil,
        refBert: MLMultiArray,
        textBert: MLMultiArray,
        vitsText: MLMultiArray? = nil,
        vitsTextLength: Int? = nil,
        refer: MLMultiArray,
        speakerFbank80: MLMultiArray,
        noiseScale: Float? = nil,
        seed: UInt64? = nil,
        samplingRNGBox: T2SSamplingRNGBox? = nil
    ) throws -> GPTSoVITSSpeakerConditionedResult {
        let semanticCodeCapacity = vits.vits.manifest.runtime.shapes.semanticCodeLen
        var t2sState = try t2s.prefill(
            prompts: prompts,
            promptLength: promptLength,
            refSeq: refSeq,
            refSeqLength: refSeqLength,
            textSeq: textSeq,
            textSeqLength: textSeqLength,
            refBert: refBert,
            textBert: textBert,
            samplingSeed: seed,
            samplingRNGBox: samplingRNGBox
        )
        let eosToken = Int32(t2s.manifest.runtime.eosToken)
        let initialSemanticToken = Int32(truncating: t2sState.lastToken[0])
        let decodedSemanticTokens = initialSemanticToken == eosToken ? [] : try t2s.decodeGreedy(
            state: &t2sState,
            limit: max(semanticCodeCapacity - 1, 0),
            stopOnEOS: true
        )
        let resolvedSemanticCodes = Self.resolveSemanticCodes(
            initialToken: initialSemanticToken,
            decodedTokens: decodedSemanticTokens,
            eosToken: eosToken,
            targetLength: semanticCodeCapacity
        )

        let codes = try vits.vits.makeInt32Array(
            shape: [1, 1, semanticCodeCapacity],
            values: resolvedSemanticCodes.paddedCodes
        )
        let resolvedVITSText = try resolveVITSText(vitsText, explicitLength: vitsTextLength)
        let vitsResult = try vits.decode(
            codes: codes,
            text: resolvedVITSText.array,
            refer: refer,
            speakerFbank80: speakerFbank80,
            noiseScale: noiseScale,
            seed: seed,
            codeLengths: [Int32(resolvedSemanticCodes.validCount)],
            textLengths: [Int32(resolvedVITSText.phoneCount)]
        )

        return GPTSoVITSSpeakerConditionedResult(
            semanticCodes: resolvedSemanticCodes.semanticCodes,
            t2sState: t2sState,
            vitsResult: vitsResult
        )
    }

    public func synthesize(
        prompts: MLMultiArray,
        promptLength: Int? = nil,
        refSeq: MLMultiArray,
        refSeqLength: Int? = nil,
        textSeq: MLMultiArray,
        textSeqLength: Int? = nil,
        refBert: MLMultiArray,
        textBert: MLMultiArray,
        vitsText: MLMultiArray? = nil,
        vitsTextLength: Int? = nil,
        referenceAudioChannels channels: [[Float]],
        referenceAudioSampleRate sampleRate: Int,
        noiseScale: Float? = nil,
        seed: UInt64? = nil
    ) throws -> GPTSoVITSSpeakerConditionedResult {
        try synthesize(
            prompts: prompts,
            promptLength: promptLength,
            refSeq: refSeq,
            refSeqLength: refSeqLength,
            textSeq: textSeq,
            textSeqLength: textSeqLength,
            refBert: refBert,
            textBert: textBert,
            vitsText: vitsText,
            vitsTextLength: vitsTextLength,
            referenceAudio: try ReferenceAudioSamples(
                channels: channels,
                sampleRate: sampleRate
            ),
            noiseScale: noiseScale,
            seed: seed
        )
    }

    public func synthesize(
        prompts: MLMultiArray,
        promptLength: Int? = nil,
        refSeq: MLMultiArray,
        refSeqLength: Int? = nil,
        textSeq: MLMultiArray,
        textSeqLength: Int? = nil,
        refBert: MLMultiArray,
        textBert: MLMultiArray,
        vitsText: MLMultiArray? = nil,
        vitsTextLength: Int? = nil,
        referenceAudio: ReferenceAudioSamples,
        noiseScale: Float? = nil,
        seed: UInt64? = nil,
        samplingRNGBox: T2SSamplingRNGBox? = nil
    ) throws -> GPTSoVITSSpeakerConditionedResult {
        let conditioning = try vits.prepareReferenceConditioning(referenceAudio: referenceAudio)
        return try synthesize(
            prompts: prompts,
            promptLength: promptLength,
            refSeq: refSeq,
            refSeqLength: refSeqLength,
            textSeq: textSeq,
            textSeqLength: textSeqLength,
            refBert: refBert,
            textBert: textBert,
            vitsText: vitsText,
            vitsTextLength: vitsTextLength,
            refer: conditioning.refer,
            speakerFbank80: conditioning.speakerFbank80,
            noiseScale: noiseScale,
            seed: seed,
            samplingRNGBox: samplingRNGBox
        )
    }

    public func makePaddedVITSText(phoneIDs: [Int32]) throws -> MLMultiArray {
        let shape = try fixedShape(for: "text", in: vits.vits.priorModel)
        guard shape.count == 2, shape[0] == 1 else {
            throw GPTSoVITSSpeakerConditionedPipelineError.invalidShape("text")
        }
        let phoneCapacity = shape[1]
        guard phoneIDs.count <= phoneCapacity else {
            throw GPTSoVITSSpeakerConditionedPipelineError.vitsTextPhoneCountExceedsCapacity(
                phoneCount: phoneIDs.count,
                capacity: phoneCapacity
            )
        }
        let padded = phoneIDs + Array(repeating: 0, count: phoneCapacity - phoneIDs.count)
        return try vits.vits.makeInt32Array(shape: shape, values: padded)
    }

    private static func resolveSemanticCodes(
        initialToken: Int32,
        decodedTokens: [Int32],
        eosToken: Int32,
        targetLength: Int
    ) -> GPTSoVITSResolvedSemanticCodes {
        var semanticCodes = [Int32]()
        semanticCodes.reserveCapacity(targetLength)
        if initialToken != eosToken, semanticCodes.count < targetLength {
            semanticCodes.append(initialToken)
        }
        for token in decodedTokens {
            if token == eosToken || semanticCodes.count >= targetLength {
                break
            }
            semanticCodes.append(token)
        }
        var paddedCodes = semanticCodes
        if paddedCodes.count < targetLength {
            paddedCodes.append(contentsOf: Array(repeating: 0, count: targetLength - paddedCodes.count))
        }
        return GPTSoVITSResolvedSemanticCodes(
            semanticCodes: semanticCodes,
            paddedCodes: paddedCodes,
            validCount: semanticCodes.count
        )
    }

    private func resolveVITSText(
        _ explicitText: MLMultiArray?,
        explicitLength: Int?
    ) throws -> GPTSoVITSResolvedVITSText {
        let expectedShape = try fixedShape(for: "text", in: vits.vits.priorModel)
        guard expectedShape.count == 2, expectedShape[0] == 1 else {
            throw GPTSoVITSSpeakerConditionedPipelineError.invalidShape("text")
        }
        let phoneCapacity = expectedShape[1]
        if let explicitText {
            try validateVITSTextShape(explicitText)
            let phoneCount = explicitLength ?? phoneCapacity
            guard phoneCount >= 0, phoneCount <= phoneCapacity else {
                throw GPTSoVITSSpeakerConditionedPipelineError.vitsTextPhoneCountExceedsCapacity(
                    phoneCount: phoneCount,
                    capacity: phoneCapacity
                )
            }
            return GPTSoVITSResolvedVITSText(array: explicitText, phoneCount: phoneCount)
        }
        return GPTSoVITSResolvedVITSText(
            array: try makePaddedVITSText(phoneIDs: []),
            phoneCount: 0
        )
    }

    private func validateVITSTextShape(_ text: MLMultiArray) throws {
        let expectedShape = try fixedShape(for: "text", in: vits.vits.priorModel)
        let actualShape = text.shape.map { Int(truncating: $0) }
        guard actualShape == expectedShape else {
            throw GPTSoVITSSpeakerConditionedPipelineError.invalidShape("text")
        }
    }

    private func fixedShape(for inputName: String, in model: MLModel) throws -> [Int] {
        guard let description = model.modelDescription.inputDescriptionsByName[inputName],
              let constraint = description.multiArrayConstraint else {
            throw GPTSoVITSSpeakerConditionedPipelineError.missingFixedShape(inputName)
        }
        let shape = constraint.shape.map { Int(truncating: $0) }
        guard !shape.isEmpty else {
            throw GPTSoVITSSpeakerConditionedPipelineError.missingFixedShape(inputName)
        }
        return shape
    }
}
