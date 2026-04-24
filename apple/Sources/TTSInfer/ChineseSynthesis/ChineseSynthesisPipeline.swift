import CoreML
import Foundation

public struct GPTSoVITSChineseSynthesisSegmentResult {
    public let preparedSegment: GPTSoVITST2SPreparedSegment
    public let synthesis: GPTSoVITSSpeakerConditionedResult
}

public struct GPTSoVITSChineseSynthesisResult {
    public let prompts: [Int32]
    public let promptExtraction: GPTSoVITSPromptSemanticResult?
    public let prompt: GPTSoVITST2SPreparedTextInput
    public let segments: [GPTSoVITSChineseSynthesisSegmentResult]

    @available(*, deprecated, renamed: "prompt")
    public var reference: GPTSoVITST2SPreparedTextInput {
        prompt
    }

    public init(
        prompts: [Int32],
        promptExtraction: GPTSoVITSPromptSemanticResult?,
        prompt: GPTSoVITST2SPreparedTextInput,
        segments: [GPTSoVITSChineseSynthesisSegmentResult]
    ) {
        self.prompts = prompts
        self.promptExtraction = promptExtraction
        self.prompt = prompt
        self.segments = segments
    }

    @available(*, deprecated, message: "Use init(prompts:promptExtraction:prompt:segments:).")
    public init(
        prompts: [Int32],
        promptExtraction: GPTSoVITSPromptSemanticResult?,
        reference: GPTSoVITST2SPreparedTextInput,
        segments: [GPTSoVITSChineseSynthesisSegmentResult]
    ) {
        self.init(
            prompts: prompts,
            promptExtraction: promptExtraction,
            prompt: reference,
            segments: segments
        )
    }
}

public enum GPTSoVITSChineseSynthesisPipelineError: LocalizedError {
    case missingFixedShape(String)
    case invalidShape(String)
    case promptCountExceedsCapacity(promptCount: Int, capacity: Int)
    case promptConditioningUnavailable

    public var errorDescription: String? {
        switch self {
        case let .missingFixedShape(name):
            return "T2S 模型输入 \(name) 没有固定 multi-array shape。"
        case let .invalidShape(name):
            return "T2S 模型输入 \(name) 的 shape 不符合当前预期。"
        case let .promptCountExceedsCapacity(promptCount, capacity):
            return "prompt token count=\(promptCount) 超过当前 T2S Core ML 导出输入容量 \(capacity)。"
        case .promptConditioningUnavailable:
            return "当前 GPTSoVITSChineseSynthesisPipeline 没有配置 cnhubert + ssl_latent prompt 提取链。"
        }
    }
}

public final class GPTSoVITSChineseSynthesisPipeline {
    public let speakerConditioned: GPTSoVITSSpeakerConditionedPipeline
    public let textPreparer: GPTSoVITST2STextPreparer
    public private(set) var promptSemantic: GPTSoVITSPromptSemanticPipeline?
    public var textPhoneBackend: (any GPTSoVITSTextPhoneBackend)? {
        get { textPreparer.textPhoneBackend }
        set { textPreparer.textPhoneBackend = newValue }
    }

    private let promptSemanticFactory: (() throws -> GPTSoVITSPromptSemanticPipeline)?

    private static func makeTextPreparer(
        t2s: T2SCoreMLDriver,
        zhBertModelURL: URL,
        tokenizerJSONURL: URL,
        g2pwBundleDirectory: URL?,
        configuration: MLModelConfiguration
    ) throws -> GPTSoVITST2STextPreparer {
        let zhBert = try GPTSoVITSRuntimeProfiler.measure("pipeline.text_preparer.zhbert.init") {
            try ZhBertCharCoreMLDriver(
                modelURL: zhBertModelURL,
                configuration: configuration
            )
        }
        let tokenizer = try GPTSoVITSRuntimeProfiler.measure("pipeline.text_preparer.tokenizer.init") {
            try GPTSoVITSZhBertTokenizer(tokenizerJSONURL: tokenizerJSONURL)
        }
        let g2pwFactory = g2pwBundleDirectory.map { bundleDirectory in
            { () throws -> (any GPTSoVITSG2PWPredicting)? in
                try GPTSoVITSRuntimeProfiler.measure("pipeline.text_preparer.g2pw.init") {
                    try G2PWCoreMLDriver(bundleDirectory: bundleDirectory, configuration: configuration)
                }
            }
        }
        return GPTSoVITSRuntimeProfiler.measure("pipeline.text_preparer.driver.init") {
            GPTSoVITST2STextPreparer(
                t2s: t2s,
                zhBert: zhBert,
                tokenizer: tokenizer,
                g2pwFactory: g2pwFactory
            )
        }
    }

    public init(
        t2sBundleDirectory: URL,
        zhBertModelURL: URL,
        tokenizerJSONURL: URL,
        vitsBundleDirectory: URL,
        g2pwBundleDirectory: URL? = nil,
        configuration: MLModelConfiguration = MLModelConfiguration()
    ) throws {
        let speakerConditioned = try GPTSoVITSRuntimeProfiler.measure("pipeline.speaker_conditioned.init") {
            try GPTSoVITSSpeakerConditionedPipeline(
                t2sBundleDirectory: t2sBundleDirectory,
                vitsBundleDirectory: vitsBundleDirectory,
                configuration: configuration
            )
        }
        self.speakerConditioned = speakerConditioned
        self.textPreparer = try GPTSoVITSRuntimeProfiler.measure("pipeline.text_preparer.init") {
            try Self.makeTextPreparer(
                t2s: speakerConditioned.t2s,
                zhBertModelURL: zhBertModelURL,
                tokenizerJSONURL: tokenizerJSONURL,
                g2pwBundleDirectory: g2pwBundleDirectory,
                configuration: configuration
            )
        }
        self.promptSemantic = nil
        self.promptSemanticFactory = nil
    }

    public init(
        t2sBundleDirectory: URL,
        zhBertModelURL: URL,
        tokenizerJSONURL: URL,
        promptBundleDirectory: URL,
        vitsBundleDirectory: URL,
        g2pwBundleDirectory: URL? = nil,
        configuration: MLModelConfiguration = MLModelConfiguration()
    ) throws {
        let speakerConditioned = try GPTSoVITSRuntimeProfiler.measure("pipeline.speaker_conditioned.init") {
            try GPTSoVITSSpeakerConditionedPipeline(
                t2sBundleDirectory: t2sBundleDirectory,
                vitsBundleDirectory: vitsBundleDirectory,
                configuration: configuration
            )
        }
        self.speakerConditioned = speakerConditioned
        self.textPreparer = try GPTSoVITSRuntimeProfiler.measure("pipeline.text_preparer.init") {
            try Self.makeTextPreparer(
                t2s: speakerConditioned.t2s,
                zhBertModelURL: zhBertModelURL,
                tokenizerJSONURL: tokenizerJSONURL,
                g2pwBundleDirectory: g2pwBundleDirectory,
                configuration: configuration
            )
        }
        self.promptSemantic = nil
        self.promptSemanticFactory = {
            try GPTSoVITSRuntimeProfiler.measure("pipeline.prompt_semantic.init") {
                try GPTSoVITSPromptSemanticPipeline(
                    bundleDirectory: promptBundleDirectory,
                    configuration: configuration
                )
            }
        }
    }

    public init(
        t2sBundleDirectory: URL,
        zhBertModelURL: URL,
        tokenizerJSONURL: URL,
        promptBundleDirectory: URL,
        speakerEncoderModelURL: URL,
        vitsBundleDirectory: URL,
        g2pwBundleDirectory: URL? = nil,
        configuration: MLModelConfiguration = MLModelConfiguration()
    ) throws {
        let speakerConditioned = try GPTSoVITSRuntimeProfiler.measure("pipeline.speaker_conditioned.init") {
            try GPTSoVITSSpeakerConditionedPipeline(
                t2sBundleDirectory: t2sBundleDirectory,
                speakerEncoderModelURL: speakerEncoderModelURL,
                vitsBundleDirectory: vitsBundleDirectory,
                configuration: configuration
            )
        }
        self.speakerConditioned = speakerConditioned
        self.textPreparer = try GPTSoVITSRuntimeProfiler.measure("pipeline.text_preparer.init") {
            try Self.makeTextPreparer(
                t2s: speakerConditioned.t2s,
                zhBertModelURL: zhBertModelURL,
                tokenizerJSONURL: tokenizerJSONURL,
                g2pwBundleDirectory: g2pwBundleDirectory,
                configuration: configuration
            )
        }
        self.promptSemantic = nil
        self.promptSemanticFactory = {
            try GPTSoVITSRuntimeProfiler.measure("pipeline.prompt_semantic.init") {
                try GPTSoVITSPromptSemanticPipeline(
                    bundleDirectory: promptBundleDirectory,
                    configuration: configuration
                )
            }
        }
    }

    public init(
        t2sBundleDirectory: URL,
        zhBertModelURL: URL,
        tokenizerJSONURL: URL,
        speakerEncoderModelURL: URL,
        vitsBundleDirectory: URL,
        g2pwBundleDirectory: URL? = nil,
        configuration: MLModelConfiguration = MLModelConfiguration()
    ) throws {
        let speakerConditioned = try GPTSoVITSRuntimeProfiler.measure("pipeline.speaker_conditioned.init") {
            try GPTSoVITSSpeakerConditionedPipeline(
                t2sBundleDirectory: t2sBundleDirectory,
                speakerEncoderModelURL: speakerEncoderModelURL,
                vitsBundleDirectory: vitsBundleDirectory,
                configuration: configuration
            )
        }
        self.speakerConditioned = speakerConditioned
        self.textPreparer = try GPTSoVITSRuntimeProfiler.measure("pipeline.text_preparer.init") {
            try Self.makeTextPreparer(
                t2s: speakerConditioned.t2s,
                zhBertModelURL: zhBertModelURL,
                tokenizerJSONURL: tokenizerJSONURL,
                g2pwBundleDirectory: g2pwBundleDirectory,
                configuration: configuration
            )
        }
        self.promptSemantic = nil
        self.promptSemanticFactory = nil
    }

    public init(
        t2sBundleDirectory: URL,
        zhBertModelURL: URL,
        tokenizerJSONURL: URL,
        cnhubertModelURL: URL,
        sslLatentExtractorModelURL: URL,
        vitsBundleDirectory: URL,
        g2pwBundleDirectory: URL? = nil,
        configuration: MLModelConfiguration = MLModelConfiguration()
    ) throws {
        let speakerConditioned = try GPTSoVITSRuntimeProfiler.measure("pipeline.speaker_conditioned.init") {
            try GPTSoVITSSpeakerConditionedPipeline(
                t2sBundleDirectory: t2sBundleDirectory,
                vitsBundleDirectory: vitsBundleDirectory,
                configuration: configuration
            )
        }
        self.speakerConditioned = speakerConditioned
        self.textPreparer = try GPTSoVITSRuntimeProfiler.measure("pipeline.text_preparer.init") {
            try Self.makeTextPreparer(
                t2s: speakerConditioned.t2s,
                zhBertModelURL: zhBertModelURL,
                tokenizerJSONURL: tokenizerJSONURL,
                g2pwBundleDirectory: g2pwBundleDirectory,
                configuration: configuration
            )
        }
        self.promptSemantic = nil
        self.promptSemanticFactory = {
            try GPTSoVITSRuntimeProfiler.measure("pipeline.prompt_semantic.init") {
                try GPTSoVITSPromptSemanticPipeline(
                    cnhubertModelURL: cnhubertModelURL,
                    sslLatentExtractorModelURL: sslLatentExtractorModelURL,
                    configuration: configuration
                )
            }
        }
    }

    public init(
        t2sBundleDirectory: URL,
        zhBertModelURL: URL,
        tokenizerJSONURL: URL,
        cnhubertModelURL: URL,
        sslLatentExtractorModelURL: URL,
        speakerEncoderModelURL: URL,
        vitsBundleDirectory: URL,
        g2pwBundleDirectory: URL? = nil,
        configuration: MLModelConfiguration = MLModelConfiguration()
    ) throws {
        let speakerConditioned = try GPTSoVITSRuntimeProfiler.measure("pipeline.speaker_conditioned.init") {
            try GPTSoVITSSpeakerConditionedPipeline(
                t2sBundleDirectory: t2sBundleDirectory,
                speakerEncoderModelURL: speakerEncoderModelURL,
                vitsBundleDirectory: vitsBundleDirectory,
                configuration: configuration
            )
        }
        self.speakerConditioned = speakerConditioned
        self.textPreparer = try GPTSoVITSRuntimeProfiler.measure("pipeline.text_preparer.init") {
            try Self.makeTextPreparer(
                t2s: speakerConditioned.t2s,
                zhBertModelURL: zhBertModelURL,
                tokenizerJSONURL: tokenizerJSONURL,
                g2pwBundleDirectory: g2pwBundleDirectory,
                configuration: configuration
            )
        }
        self.promptSemantic = nil
        self.promptSemanticFactory = {
            try GPTSoVITSRuntimeProfiler.measure("pipeline.prompt_semantic.init") {
                try GPTSoVITSPromptSemanticPipeline(
                    cnhubertModelURL: cnhubertModelURL,
                    sslLatentExtractorModelURL: sslLatentExtractorModelURL,
                    configuration: configuration
                )
            }
        }
    }

    public func synthesize(
        prompts: [Int32],
        promptText: String,
        targetText: String,
        referenceAudio: ReferenceAudioSamples,
        language: GPTSoVITSTextLanguage = .zh,
        splitMethod: GPTSoVITSTextSplitMethod = .cut5,
        noiseScale: Float? = nil,
        seed: UInt64? = nil
    ) throws -> GPTSoVITSChineseSynthesisResult {
        let prepared = try textPreparer.preparePromptAndTargets(
            promptText: promptText,
            targetText: targetText,
            language: language,
            splitMethod: splitMethod
        )
        let promptsArray = try makePaddedPrompts(prompts)
        let conditioning = try speakerConditioned.vits.prepareReferenceConditioning(referenceAudio: referenceAudio)
        let t2sSamplingRNGBox = seed.map(T2SSamplingRNGBox.init(seed:))

        let synthesizedSegments = try prepared.targets.map { segment in
            let vitsText = try speakerConditioned.makePaddedVITSText(phoneIDs: segment.input.phoneIDs)
            return GPTSoVITSChineseSynthesisSegmentResult(
                preparedSegment: segment,
                synthesis: try speakerConditioned.synthesize(
                    prompts: promptsArray,
                    promptLength: prompts.count,
                    refSeq: prepared.prompt.seq,
                    refSeqLength: prepared.prompt.phoneCount,
                    textSeq: segment.input.seq,
                    textSeqLength: segment.input.phoneCount,
                    refBert: prepared.prompt.bert,
                    textBert: segment.input.bert,
                    vitsText: vitsText,
                    vitsTextLength: segment.input.phoneIDs.count,
                    refer: conditioning.refer,
                    speakerFbank80: conditioning.speakerFbank80,
                    noiseScale: noiseScale,
                    seed: seed,
                    samplingRNGBox: t2sSamplingRNGBox
                )
            )
        }

        return GPTSoVITSChineseSynthesisResult(
            prompts: prompts,
            promptExtraction: nil,
            prompt: prepared.prompt,
            segments: synthesizedSegments
        )
    }

    @available(*, deprecated, message: "Use synthesize(prompts:promptText:targetText:referenceAudio:language:splitMethod:noiseScale:seed:).")
    public func synthesize(
        prompts: [Int32],
        referenceText: String,
        targetText: String,
        referenceAudio: ReferenceAudioSamples,
        language: GPTSoVITSTextLanguage = .zh,
        splitMethod: GPTSoVITSTextSplitMethod = .cut5,
        noiseScale: Float? = nil,
        seed: UInt64? = nil
    ) throws -> GPTSoVITSChineseSynthesisResult {
        try synthesize(
            prompts: prompts,
            promptText: referenceText,
            targetText: targetText,
            referenceAudio: referenceAudio,
            language: language,
            splitMethod: splitMethod,
            noiseScale: noiseScale,
            seed: seed
        )
    }

    public func synthesize(
        promptText: String,
        targetText: String,
        referenceAudio: ReferenceAudioSamples,
        language: GPTSoVITSTextLanguage = .zh,
        splitMethod: GPTSoVITSTextSplitMethod = .cut5,
        noiseScale: Float? = nil,
        seed: UInt64? = nil
    ) throws -> GPTSoVITSChineseSynthesisResult {
        let promptSemantic = try requirePromptSemantic()
        let promptResult = try promptSemantic.extractPrompts(referenceAudio: referenceAudio)
        let synthesized = try synthesize(
            prompts: promptResult.prompts,
            promptText: promptText,
            targetText: targetText,
            referenceAudio: referenceAudio,
            language: language,
            splitMethod: splitMethod,
            noiseScale: noiseScale,
            seed: seed
        )
        return GPTSoVITSChineseSynthesisResult(
            prompts: synthesized.prompts,
            promptExtraction: promptResult,
            prompt: synthesized.prompt,
            segments: synthesized.segments
        )
    }

    @available(*, deprecated, message: "Use synthesize(promptText:targetText:referenceAudio:language:splitMethod:noiseScale:seed:).")
    public func synthesize(
        referenceText: String,
        targetText: String,
        referenceAudio: ReferenceAudioSamples,
        language: GPTSoVITSTextLanguage = .zh,
        splitMethod: GPTSoVITSTextSplitMethod = .cut5,
        noiseScale: Float? = nil,
        seed: UInt64? = nil
    ) throws -> GPTSoVITSChineseSynthesisResult {
        try synthesize(
            promptText: referenceText,
            targetText: targetText,
            referenceAudio: referenceAudio,
            language: language,
            splitMethod: splitMethod,
            noiseScale: noiseScale,
            seed: seed
        )
    }

    public func synthesize(
        promptText: String,
        targetText: String,
        promptReferenceAudio: ReferenceAudioSamples,
        conditioningReferenceAudio: ReferenceAudioSamples,
        language: GPTSoVITSTextLanguage = .zh,
        splitMethod: GPTSoVITSTextSplitMethod = .cut5,
        noiseScale: Float? = nil,
        seed: UInt64? = nil
    ) throws -> GPTSoVITSChineseSynthesisResult {
        let promptSemantic = try requirePromptSemantic()
        let promptResult = try promptSemantic.extractPrompts(referenceAudio: promptReferenceAudio)
        let synthesized = try synthesize(
            prompts: promptResult.prompts,
            promptText: promptText,
            targetText: targetText,
            referenceAudio: conditioningReferenceAudio,
            language: language,
            splitMethod: splitMethod,
            noiseScale: noiseScale,
            seed: seed
        )
        return GPTSoVITSChineseSynthesisResult(
            prompts: synthesized.prompts,
            promptExtraction: promptResult,
            prompt: synthesized.prompt,
            segments: synthesized.segments
        )
    }

    @available(*, deprecated, message: "Use synthesize(promptText:targetText:promptReferenceAudio:conditioningReferenceAudio:language:splitMethod:noiseScale:seed:).")
    public func synthesize(
        referenceText: String,
        targetText: String,
        promptReferenceAudio: ReferenceAudioSamples,
        conditioningReferenceAudio: ReferenceAudioSamples,
        language: GPTSoVITSTextLanguage = .zh,
        splitMethod: GPTSoVITSTextSplitMethod = .cut5,
        noiseScale: Float? = nil,
        seed: UInt64? = nil
    ) throws -> GPTSoVITSChineseSynthesisResult {
        try synthesize(
            promptText: referenceText,
            targetText: targetText,
            promptReferenceAudio: promptReferenceAudio,
            conditioningReferenceAudio: conditioningReferenceAudio,
            language: language,
            splitMethod: splitMethod,
            noiseScale: noiseScale,
            seed: seed
        )
    }

    public func synthesize(
        promptText: String,
        targetText: String,
        referenceAudioChannels channels: [[Float]],
        referenceAudioSampleRate sampleRate: Int,
        language: GPTSoVITSTextLanguage = .zh,
        splitMethod: GPTSoVITSTextSplitMethod = .cut5,
        noiseScale: Float? = nil,
        seed: UInt64? = nil
    ) throws -> GPTSoVITSChineseSynthesisResult {
        try synthesize(
            promptText: promptText,
            targetText: targetText,
            referenceAudio: try ReferenceAudioSamples(
                channels: channels,
                sampleRate: sampleRate
            ),
            language: language,
            splitMethod: splitMethod,
            noiseScale: noiseScale,
            seed: seed
        )
    }

    @available(*, deprecated, message: "Use synthesize(promptText:targetText:referenceAudioChannels:referenceAudioSampleRate:language:splitMethod:noiseScale:seed:).")
    public func synthesize(
        referenceText: String,
        targetText: String,
        referenceAudioChannels channels: [[Float]],
        referenceAudioSampleRate sampleRate: Int,
        language: GPTSoVITSTextLanguage = .zh,
        splitMethod: GPTSoVITSTextSplitMethod = .cut5,
        noiseScale: Float? = nil,
        seed: UInt64? = nil
    ) throws -> GPTSoVITSChineseSynthesisResult {
        try synthesize(
            promptText: referenceText,
            targetText: targetText,
            referenceAudioChannels: channels,
            referenceAudioSampleRate: sampleRate,
            language: language,
            splitMethod: splitMethod,
            noiseScale: noiseScale,
            seed: seed
        )
    }

    public func synthesize(
        promptText: String,
        targetText: String,
        promptReferenceAudioChannels promptChannels: [[Float]],
        promptReferenceAudioSampleRate promptSampleRate: Int,
        conditioningReferenceAudioChannels conditioningChannels: [[Float]],
        conditioningReferenceAudioSampleRate conditioningSampleRate: Int,
        language: GPTSoVITSTextLanguage = .zh,
        splitMethod: GPTSoVITSTextSplitMethod = .cut5,
        noiseScale: Float? = nil,
        seed: UInt64? = nil
    ) throws -> GPTSoVITSChineseSynthesisResult {
        try synthesize(
            promptText: promptText,
            targetText: targetText,
            promptReferenceAudio: try ReferenceAudioSamples(
                channels: promptChannels,
                sampleRate: promptSampleRate
            ),
            conditioningReferenceAudio: try ReferenceAudioSamples(
                channels: conditioningChannels,
                sampleRate: conditioningSampleRate
            ),
            language: language,
            splitMethod: splitMethod,
            noiseScale: noiseScale,
            seed: seed
        )
    }

    private func requirePromptSemantic() throws -> GPTSoVITSPromptSemanticPipeline {
        if let promptSemantic {
            return promptSemantic
        }
        guard let promptSemanticFactory else {
            throw GPTSoVITSChineseSynthesisPipelineError.promptConditioningUnavailable
        }
        let promptSemantic = try promptSemanticFactory()
        self.promptSemantic = promptSemantic
        return promptSemantic
    }

    @available(*, deprecated, message: "Use synthesize(promptText:targetText:promptReferenceAudioChannels:promptReferenceAudioSampleRate:conditioningReferenceAudioChannels:conditioningReferenceAudioSampleRate:language:splitMethod:noiseScale:seed:).")
    public func synthesize(
        referenceText: String,
        targetText: String,
        promptReferenceAudioChannels promptChannels: [[Float]],
        promptReferenceAudioSampleRate promptSampleRate: Int,
        conditioningReferenceAudioChannels conditioningChannels: [[Float]],
        conditioningReferenceAudioSampleRate conditioningSampleRate: Int,
        language: GPTSoVITSTextLanguage = .zh,
        splitMethod: GPTSoVITSTextSplitMethod = .cut5,
        noiseScale: Float? = nil,
        seed: UInt64? = nil
    ) throws -> GPTSoVITSChineseSynthesisResult {
        try synthesize(
            promptText: referenceText,
            targetText: targetText,
            promptReferenceAudioChannels: promptChannels,
            promptReferenceAudioSampleRate: promptSampleRate,
            conditioningReferenceAudioChannels: conditioningChannels,
            conditioningReferenceAudioSampleRate: conditioningSampleRate,
            language: language,
            splitMethod: splitMethod,
            noiseScale: noiseScale,
            seed: seed
        )
    }

    public func synthesize(
        prompts: [Int32],
        promptText: String,
        targetText: String,
        referenceAudioChannels channels: [[Float]],
        referenceAudioSampleRate sampleRate: Int,
        language: GPTSoVITSTextLanguage = .zh,
        splitMethod: GPTSoVITSTextSplitMethod = .cut5,
        noiseScale: Float? = nil,
        seed: UInt64? = nil
    ) throws -> GPTSoVITSChineseSynthesisResult {
        try synthesize(
            prompts: prompts,
            promptText: promptText,
            targetText: targetText,
            referenceAudio: try ReferenceAudioSamples(
                channels: channels,
                sampleRate: sampleRate
            ),
            language: language,
            splitMethod: splitMethod,
            noiseScale: noiseScale,
            seed: seed
        )
    }

    @available(*, deprecated, message: "Use synthesize(prompts:promptText:targetText:referenceAudioChannels:referenceAudioSampleRate:language:splitMethod:noiseScale:seed:).")
    public func synthesize(
        prompts: [Int32],
        referenceText: String,
        targetText: String,
        referenceAudioChannels channels: [[Float]],
        referenceAudioSampleRate sampleRate: Int,
        language: GPTSoVITSTextLanguage = .zh,
        splitMethod: GPTSoVITSTextSplitMethod = .cut5,
        noiseScale: Float? = nil,
        seed: UInt64? = nil
    ) throws -> GPTSoVITSChineseSynthesisResult {
        try synthesize(
            prompts: prompts,
            promptText: referenceText,
            targetText: targetText,
            referenceAudioChannels: channels,
            referenceAudioSampleRate: sampleRate,
            language: language,
            splitMethod: splitMethod,
            noiseScale: noiseScale,
            seed: seed
        )
    }

    private func makePaddedPrompts(_ prompts: [Int32]) throws -> MLMultiArray {
        if speakerConditioned.t2s.manifest.runtime.shapes?.promptLenRange != nil {
            let shape = [1, max(prompts.count, 1)]
            let values = prompts + Array(repeating: 0, count: shape[1] - prompts.count)
            return try speakerConditioned.t2s.makeInt32Array(shape: shape, values: values)
        }
        let shape = try fixedShape(for: "prompts", in: speakerConditioned.t2s.prefillModel)
        guard shape.count == 2, shape[0] == 1 else {
            throw GPTSoVITSChineseSynthesisPipelineError.invalidShape("prompts")
        }
        let capacity = shape[1]
        guard prompts.count <= capacity else {
            throw GPTSoVITSChineseSynthesisPipelineError.promptCountExceedsCapacity(
                promptCount: prompts.count,
                capacity: capacity
            )
        }
        let padded = prompts + Array(repeating: 0, count: capacity - prompts.count)
        return try speakerConditioned.t2s.makeInt32Array(shape: shape, values: padded)
    }

    private func fixedShape(for inputName: String, in model: MLModel) throws -> [Int] {
        guard let description = model.modelDescription.inputDescriptionsByName[inputName],
              let constraint = description.multiArrayConstraint else {
            throw GPTSoVITSChineseSynthesisPipelineError.missingFixedShape(inputName)
        }
        let shape = constraint.shape.map { Int(truncating: $0) }
        guard !shape.isEmpty else {
            throw GPTSoVITSChineseSynthesisPipelineError.missingFixedShape(inputName)
        }
        return shape
    }
}
