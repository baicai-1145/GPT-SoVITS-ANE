import CoreML
import Foundation

public struct T2SBundleManifest: Decodable {
    public struct Artifact: Decodable {
        public let target: String
        public let filename: String
        public let path: String
    }

    public struct Artifacts: Decodable {
        public let prefill: Artifact?
        public let prefillPrepare: Artifact?
        public let prefillCore: Artifact?
        public let decodeStep: Artifact?
        public let decodePrepare: Artifact?
        public let decodeCore: Artifact?

        private enum CodingKeys: String, CodingKey {
            case prefill
            case prefillPrepare = "prefill_prepare"
            case prefillCore = "prefill_core"
            case decodeStep = "decode_step"
            case decodePrepare = "decode_prepare"
            case decodeCore = "decode_core"
        }
    }

    public struct Runtime: Decodable {
        public struct Shapes: Decodable {
            public struct ShapeRange: Decodable {
                public let lowerBound: Int
                public let upperBound: Int

                private enum CodingKeys: String, CodingKey {
                    case lowerBound = "lower_bound"
                    case upperBound = "upper_bound"
                }
            }

            public let promptLen: Int?
            public let promptLenRange: ShapeRange?
            public let refPhoneLen: Int?
            public let refPhoneLenRange: ShapeRange?
            public let textPhoneLen: Int?
            public let textPhoneLenRange: ShapeRange?

            private enum CodingKeys: String, CodingKey {
                case promptLen = "prompt_len"
                case promptLenRange = "prompt_len_range"
                case refPhoneLen = "ref_phone_len"
                case refPhoneLenRange = "ref_phone_len_range"
                case textPhoneLen = "text_phone_len"
                case textPhoneLenRange = "text_phone_len_range"
            }
        }

        public struct SamplingDefaults: Decodable {
            public let mode: String?
            public let topK: Int?
            public let topP: Double?
            public let temperature: Double?
            public let repetitionPenalty: Double?

            private enum CodingKeys: String, CodingKey {
                case mode
                case topK = "top_k"
                case topP = "top_p"
                case temperature
                case repetitionPenalty = "repetition_penalty"
            }
        }

        public let eosToken: Int
        public let maxDecodeSteps: Int
        public let prefillExportMode: String?
        public let decodeExportMode: String?
        public let samplingDefaults: SamplingDefaults?
        public let shapes: Shapes?

        private enum CodingKeys: String, CodingKey {
            case eosToken = "eos_token"
            case maxDecodeSteps = "max_decode_steps"
            case prefillExportMode = "prefill_export_mode"
            case decodeExportMode = "decode_export_mode"
            case samplingDefaults = "sampling_defaults"
            case shapes
        }
    }

    public let bundleType: String
    public let bundleDir: String
    public let artifacts: Artifacts
    public let runtime: Runtime

    private enum CodingKeys: String, CodingKey {
        case bundleType = "bundle_type"
        case bundleDir = "bundle_dir"
        case artifacts
        case runtime
    }
}

public struct T2SDecodeState {
    public var lastToken: MLMultiArray
    public var cacheLen: MLMultiArray
    public var nextPosition: MLMultiArray
    public var kCache: MLMultiArray
    public var vCache: MLMultiArray
    public var generatedTokens: [Int32]
    public var promptTokenCount: Int
    public var samplingRNGState: UInt64?
    public var samplingRNGBox: T2SSamplingRNGBox?
}

public final class T2SSamplingRNGBox {
    private static let stateCount = 624
    private static let middleWord = 397
    private static let matrixA: UInt32 = 0x9908_B0DF
    private static let upperMask: UInt32 = 0x8000_0000
    private static let lowerMask: UInt32 = 0x7FFF_FFFF

    private var state: [UInt32]
    private var index: Int

    public init(seed: UInt64) {
        self.state = Array(repeating: 0, count: Self.stateCount)
        self.state[0] = UInt32(truncatingIfNeeded: seed)
        if Self.stateCount > 1 {
            for offset in 1..<Self.stateCount {
                let previous = self.state[offset - 1]
                self.state[offset] = 1812433253 &* (previous ^ (previous >> 30)) &+ UInt32(offset)
            }
        }
        self.index = Self.stateCount
    }

    public func nextUnitDouble53() -> Double {
        let combined = (UInt64(nextUInt32()) << 32) | UInt64(nextUInt32())
        let mantissaMask: UInt64 = (UInt64(1) << 53) - 1
        return Double(combined & mantissaMask) / 9007199254740992.0
    }

    private func nextUInt32() -> UInt32 {
        if index >= Self.stateCount {
            twist()
        }
        var value = state[index]
        index += 1
        value ^= value >> 11
        value ^= (value << 7) & 0x9D2C_5680
        value ^= (value << 15) & 0xEFC6_0000
        value ^= value >> 18
        return value
    }

    private func twist() {
        for offset in 0..<Self.stateCount {
            let combined = (state[offset] & Self.upperMask) |
                (state[(offset + 1) % Self.stateCount] & Self.lowerMask)
            var nextValue = state[(offset + Self.middleWord) % Self.stateCount] ^ (combined >> 1)
            if (combined & 1) != 0 {
                nextValue ^= Self.matrixA
            }
            state[offset] = nextValue
        }
        index = 0
    }
}

private struct SplitMix64: RandomNumberGenerator {
    var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}

private struct T2SSamplingParameters {
    let mode: String
    let topK: Int?
    let topP: Double?
    let temperature: Double
    let repetitionPenalty: Double
}

private struct T2SSamplingResult {
    let sampledToken: Int32
    let greedyToken: Int32
    let shouldStop: Bool
}

private struct T2SDebugTopEntries {
    let tokens: [Int]
    let values: [Float]
}

public final class T2SCoreMLDriver {
    public let manifest: T2SBundleManifest
    public let prefillModel: MLModel
    public let decodeModel: MLModel
    public let usesSplitPrefill: Bool
    public let usesSplitDecode: Bool
    private let prefillCoreModel: MLModel?
    private let decodeCoreModel: MLModel?

    public init(bundleDirectory: URL, configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        let manifestURL = bundleDirectory.appendingPathComponent("manifest.json")
        let manifestData = try GPTSoVITSRuntimeProfiler.measure("t2s.init.manifest.read") {
            try Data(contentsOf: manifestURL)
        }
        let decoder = JSONDecoder()
        self.manifest = try GPTSoVITSRuntimeProfiler.measure("t2s.init.manifest.decode") {
            try decoder.decode(T2SBundleManifest.self, from: manifestData)
        }

        let resolvedUsesSplitPrefill: Bool
        switch (
            manifest.artifacts.prefill != nil,
            manifest.artifacts.prefillPrepare != nil && manifest.artifacts.prefillCore != nil
        ) {
        case (true, false):
            resolvedUsesSplitPrefill = false
        case (false, true):
            resolvedUsesSplitPrefill = true
        case (false, false):
            throw NSError(domain: "T2SCoreMLDriver", code: 10, userInfo: [
                NSLocalizedDescriptionKey: "T2S bundle must contain either 'prefill' or both 'prefill_prepare' and 'prefill_core' artifacts."
            ])
        case (true, true):
            throw NSError(domain: "T2SCoreMLDriver", code: 16, userInfo: [
                NSLocalizedDescriptionKey: "T2S bundle cannot contain both monolithic 'prefill' and split 'prefill_prepare/prefill_core' artifacts."
            ])
        }

        let resolvedUsesSplitDecode: Bool
        switch (
            manifest.artifacts.decodeStep != nil,
            manifest.artifacts.decodePrepare != nil && manifest.artifacts.decodeCore != nil
        ) {
        case (true, false):
            resolvedUsesSplitDecode = false
        case (false, true):
            resolvedUsesSplitDecode = true
        case (false, false):
            throw NSError(domain: "T2SCoreMLDriver", code: 11, userInfo: [
                NSLocalizedDescriptionKey: "T2S bundle must contain either 'decode_step' or both 'decode_prepare' and 'decode_core' artifacts."
            ])
        case (true, true):
            throw NSError(domain: "T2SCoreMLDriver", code: 17, userInfo: [
                NSLocalizedDescriptionKey: "T2S bundle cannot contain both monolithic 'decode_step' and split 'decode_prepare/decode_core' artifacts."
            ])
        }

        self.usesSplitPrefill = resolvedUsesSplitPrefill
        self.usesSplitDecode = resolvedUsesSplitDecode

        if let prefillArtifact = manifest.artifacts.prefill {
            let prefillURL = bundleDirectory.appendingPathComponent(prefillArtifact.filename)
            self.prefillModel = try GPTSoVITSRuntimeProfiler.measure("t2s.init.prefill.load") {
                try Self.loadModel(at: prefillURL, configuration: configuration)
            }
            self.prefillCoreModel = nil
        } else if let prepareArtifact = manifest.artifacts.prefillPrepare,
                  let coreArtifact = manifest.artifacts.prefillCore {
            let prepareURL = bundleDirectory.appendingPathComponent(prepareArtifact.filename)
            let coreURL = bundleDirectory.appendingPathComponent(coreArtifact.filename)
            self.prefillModel = try GPTSoVITSRuntimeProfiler.measure("t2s.init.prefill_prepare.load") {
                try Self.loadModel(at: prepareURL, configuration: configuration)
            }
            self.prefillCoreModel = try GPTSoVITSRuntimeProfiler.measure("t2s.init.prefill_core.load") {
                try Self.loadModel(at: coreURL, configuration: configuration)
            }
        } else {
            throw NSError(domain: "T2SCoreMLDriver", code: 10, userInfo: [
                NSLocalizedDescriptionKey: "T2S bundle must contain either 'prefill' or both 'prefill_prepare' and 'prefill_core' artifacts."
            ])
        }
        if let decodeArtifact = manifest.artifacts.decodeStep {
            let decodeURL = bundleDirectory.appendingPathComponent(decodeArtifact.filename)
            self.decodeModel = try GPTSoVITSRuntimeProfiler.measure("t2s.init.decode_step.load") {
                try Self.loadModel(at: decodeURL, configuration: configuration)
            }
            self.decodeCoreModel = nil
        } else if let decodePrepareArtifact = manifest.artifacts.decodePrepare,
                  let decodeCoreArtifact = manifest.artifacts.decodeCore {
            let decodePrepareURL = bundleDirectory.appendingPathComponent(decodePrepareArtifact.filename)
            let decodeCoreURL = bundleDirectory.appendingPathComponent(decodeCoreArtifact.filename)
            self.decodeModel = try GPTSoVITSRuntimeProfiler.measure("t2s.init.decode_prepare.load") {
                try Self.loadModel(at: decodePrepareURL, configuration: configuration)
            }
            self.decodeCoreModel = try GPTSoVITSRuntimeProfiler.measure("t2s.init.decode_core.load") {
                try Self.loadModel(at: decodeCoreURL, configuration: configuration)
            }
        } else {
            throw NSError(domain: "T2SCoreMLDriver", code: 11, userInfo: [
                NSLocalizedDescriptionKey: "T2S bundle must contain either 'decode_step' or both 'decode_prepare' and 'decode_core' artifacts."
            ])
        }
    }

    public func makeInt32Array(shape: [Int], values: [Int32]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map(NSNumber.init(value:)), dataType: .int32)
        guard array.count == values.count else {
            throw NSError(domain: "T2SCoreMLDriver", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Value count \(values.count) does not match shape element count \(array.count)."
            ])
        }
        for (index, value) in values.enumerated() {
            array[index] = NSNumber(value: value)
        }
        return array
    }

    public func prefill(
        prompts: MLMultiArray,
        promptLength: Int? = nil,
        refSeq: MLMultiArray,
        refSeqLength: Int? = nil,
        textSeq: MLMultiArray,
        textSeqLength: Int? = nil,
        refBert: MLMultiArray,
        textBert: MLMultiArray,
        samplingSeed: UInt64? = nil,
        samplingRNGBox: T2SSamplingRNGBox? = nil
    ) throws -> T2SDecodeState {
        let promptShape = try requireShape(prompts, rank: 2, name: "prompts")
        let refSeqShape = try requireShape(refSeq, rank: 2, name: "ref_seq")
        let textSeqShape = try requireShape(textSeq, rank: 2, name: "text_seq")
        let resolvedPromptLength = promptLength ?? promptShape[1]
        let shouldMaterializePrefillInputs = usesSplitPrefill &&
            ProcessInfo.processInfo.environment["GPTSOVITS_T2S_MATERIALIZE_PREFILL_INPUTS"] == "1"
        let preparedPrompts = shouldMaterializePrefillInputs ? try materializeArray(prompts) : prompts
        let preparedRefSeq = shouldMaterializePrefillInputs ? try materializeArray(refSeq) : refSeq
        let preparedTextSeq = shouldMaterializePrefillInputs ? try materializeArray(textSeq) : textSeq
        let preparedRefBert = shouldMaterializePrefillInputs ? try materializeArray(refBert) : refBert
        let preparedTextBert = shouldMaterializePrefillInputs ? try materializeArray(textBert) : textBert
        var dictionary: [String: MLFeatureValue] = [
            "prompts": MLFeatureValue(multiArray: preparedPrompts),
            "ref_seq": MLFeatureValue(multiArray: preparedRefSeq),
            "text_seq": MLFeatureValue(multiArray: preparedTextSeq),
            "ref_bert": MLFeatureValue(multiArray: preparedRefBert),
            "text_bert": MLFeatureValue(multiArray: preparedTextBert),
        ]
        if prefillModel.modelDescription.inputDescriptionsByName["prompt_length"] != nil {
            dictionary["prompt_length"] = MLFeatureValue(
                multiArray: try resolveLengthArray(
                    length: promptLength,
                    defaultLength: promptShape[1],
                    capacity: promptShape[1]
                )
            )
        }
        if prefillModel.modelDescription.inputDescriptionsByName["ref_seq_length"] != nil {
            dictionary["ref_seq_length"] = MLFeatureValue(
                multiArray: try resolveLengthArray(
                    length: refSeqLength,
                    defaultLength: refSeqShape[1],
                    capacity: refSeqShape[1]
                )
            )
        }
        if prefillModel.modelDescription.inputDescriptionsByName["text_seq_length"] != nil {
            dictionary["text_seq_length"] = MLFeatureValue(
                multiArray: try resolveLengthArray(
                    length: textSeqLength,
                    defaultLength: textSeqShape[1],
                    capacity: textSeqShape[1]
                )
            )
        }
        if ProcessInfo.processInfo.environment["GPTSOVITS_DEBUG_T2S_INPUTS"] == "1" {
            emitInputDebug(
                prefix: "prefill_inputs",
                arrays: [
                    "prompts": preparedPrompts,
                    "ref_seq": preparedRefSeq,
                    "text_seq": preparedTextSeq,
                    "ref_bert": preparedRefBert,
                    "text_bert": preparedTextBert,
                ]
            )
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: dictionary)
        let output: MLFeatureProvider
        if let prefillCoreModel {
            let prepareOutput = try prefillModel.prediction(from: provider)
            if ProcessInfo.processInfo.environment["GPTSOVITS_DEBUG_T2S_PREPARE"] == "1" {
                emitPrepareDebug(prefix: "prefill_prepare", provider: prepareOutput)
            }
            var coreDictionary: [String: MLFeatureValue] = [
                "xy_pos": MLFeatureValue(
                    multiArray: try materializeArray(try requireArray(named: "xy_pos", from: prepareOutput))
                ),
                "prompt_attn_mask": MLFeatureValue(
                    multiArray: try materializeArray(try requireArray(named: "prompt_attn_mask", from: prepareOutput))
                ),
            ]
            if prefillCoreModel.modelDescription.inputDescriptionsByName["active_src_len"] != nil,
               let activeSrcLen = prepareOutput.featureValue(for: "active_src_len")?.multiArrayValue {
                coreDictionary["active_src_len"] = MLFeatureValue(
                    multiArray: try materializeArray(activeSrcLen)
                )
            } else if prefillCoreModel.modelDescription.inputDescriptionsByName["prompt_len"] != nil {
                coreDictionary["prompt_len"] = MLFeatureValue(
                    multiArray: try materializeArray(try requireArray(named: "prompt_len", from: prepareOutput))
                )
            }
            if prefillCoreModel.modelDescription.inputDescriptionsByName["position_seed"] != nil {
                if let nextPosition = prepareOutput.featureValue(for: "next_position")?.multiArrayValue {
                    coreDictionary["position_seed"] = MLFeatureValue(
                        multiArray: try materializeArray(nextPosition)
                    )
                } else if let promptLen = prepareOutput.featureValue(for: "prompt_len")?.multiArrayValue {
                    coreDictionary["position_seed"] = MLFeatureValue(
                        multiArray: try materializeArray(promptLen)
                    )
                } else {
                    coreDictionary["position_seed"] = MLFeatureValue(
                        multiArray: try resolveLengthArray(
                            length: promptLength,
                            defaultLength: promptShape[1],
                            capacity: promptShape[1]
                        )
                    )
                }
            }
            let coreProvider = try MLDictionaryFeatureProvider(dictionary: coreDictionary)
            output = try prefillCoreModel.prediction(from: coreProvider)
            if ProcessInfo.processInfo.environment["GPTSOVITS_DEBUG_T2S_RAW"] == "1" {
                emitRawDebug(prefix: "prefill", provider: output, generatedTokenCount: 0)
            }
        } else {
            output = try prefillModel.prediction(from: provider)
            if ProcessInfo.processInfo.environment["GPTSOVITS_DEBUG_T2S_RAW"] == "1" {
                emitRawDebug(prefix: "prefill", provider: output, generatedTokenCount: 0)
            }
        }
        let logits = try requireArray(named: "logits", from: output)
        let promptTokens = try int32Values(from: prompts, activeCount: resolvedPromptLength)
        let sampling = resolvedSamplingParameters()
        var samplingRNGState = samplingSeed
        let resolvedSamplingRNGBox = samplingRNGBox ?? samplingSeed.map(T2SSamplingRNGBox.init(seed:))
        let prefillSampling: T2SSamplingResult
        if sampling.mode == "coreml_greedy_token",
           prefersModelGreedyToken(),
           let modelSampledToken = sampledToken(from: output) {
            let modelEOS = eosReached(from: output) ?? (modelSampledToken == Int32(manifest.runtime.eosToken))
            prefillSampling = T2SSamplingResult(
                sampledToken: modelSampledToken,
                greedyToken: modelSampledToken,
                shouldStop: modelEOS
            )
        } else {
            let hostSampling = sampleTokenFromLogits(
                logits,
                previousTokens: promptTokens,
                generatedTokenCount: 0,
                sampling: sampling,
                rngState: &samplingRNGState,
                rngBox: resolvedSamplingRNGBox
            )
            prefillSampling = T2SSamplingResult(
                sampledToken: hostSampling.sampledToken,
                greedyToken: hostSampling.greedyToken,
                shouldStop: hostSampling.shouldStop || (eosReached(from: output) ?? false)
            )
        }
        let acceptedInitialToken = prefillSampling.shouldStop ? Int32(manifest.runtime.eosToken) : prefillSampling.sampledToken
        let lastToken = try makeInt32Array(shape: [1, 1], values: [acceptedInitialToken])
        var generatedTokens = promptTokens
        if !prefillSampling.shouldStop {
            generatedTokens.append(acceptedInitialToken)
        }
        return T2SDecodeState(
            lastToken: lastToken,
            cacheLen: try materializeArray(try requireArray(named: "cache_len", from: output)),
            nextPosition: try materializeArray(try requireArray(named: "next_position", from: output)),
            kCache: try materializeArray(try requireArray(named: "k_cache", from: output)),
            vCache: try materializeArray(try requireArray(named: "v_cache", from: output)),
            generatedTokens: generatedTokens,
            promptTokenCount: promptTokens.count,
            samplingRNGState: samplingRNGState,
            samplingRNGBox: resolvedSamplingRNGBox
        )
    }

    public func decodeStep(state: inout T2SDecodeState) throws -> MLMultiArray {
        return try GPTSoVITSRuntimeProfiler.measure("t2s.decode.step.total") {
            let output: MLFeatureProvider
            if let decodeCoreModel {
                let prepareProvider = try GPTSoVITSRuntimeProfiler.measure("t2s.decode.prepare.provider") {
                    try MLDictionaryFeatureProvider(dictionary: [
                        "last_token": MLFeatureValue(multiArray: state.lastToken),
                        "position_index": MLFeatureValue(multiArray: state.nextPosition),
                    ])
                }
                let prepareOutput = try GPTSoVITSRuntimeProfiler.measure("t2s.decode.prepare.prediction") {
                    try decodeModel.prediction(from: prepareProvider)
                }
                if ProcessInfo.processInfo.environment["GPTSOVITS_DEBUG_T2S_PREPARE"] == "1",
                   state.generatedTokens.count - state.promptTokenCount < 4 {
                    emitPrepareDebug(
                        prefix: "decode_prepare_\(state.generatedTokens.count - state.promptTokenCount + 1)",
                        provider: prepareOutput
                    )
                }
                let xyPos = try GPTSoVITSRuntimeProfiler.measure("t2s.decode.prepare.xy_pos_materialize") {
                    try materializeArray(try requireArray(named: "xy_pos", from: prepareOutput))
                }
                let coreProvider = try GPTSoVITSRuntimeProfiler.measure("t2s.decode.core.provider") {
                    try MLDictionaryFeatureProvider(dictionary: [
                        "xy_pos": MLFeatureValue(multiArray: xyPos),
                        "position_index": MLFeatureValue(multiArray: state.nextPosition),
                        "cache_len": MLFeatureValue(multiArray: state.cacheLen),
                        "k_cache": MLFeatureValue(multiArray: state.kCache),
                        "v_cache": MLFeatureValue(multiArray: state.vCache),
                    ])
                }
                output = try GPTSoVITSRuntimeProfiler.measure("t2s.decode.core.prediction") {
                    try decodeCoreModel.prediction(from: coreProvider)
                }
                if ProcessInfo.processInfo.environment["GPTSOVITS_DEBUG_T2S_RAW"] == "1",
                   state.generatedTokens.count - state.promptTokenCount < 4 {
                    emitRawDebug(
                        prefix: "decode_step_\(state.generatedTokens.count - state.promptTokenCount + 1)",
                        provider: output,
                        generatedTokenCount: state.generatedTokens.count - state.promptTokenCount
                    )
                }
            } else {
                let provider = try GPTSoVITSRuntimeProfiler.measure("t2s.decode.provider") {
                    try MLDictionaryFeatureProvider(dictionary: [
                        "last_token": MLFeatureValue(multiArray: state.lastToken),
                        "position_index": MLFeatureValue(multiArray: state.nextPosition),
                        "cache_len": MLFeatureValue(multiArray: state.cacheLen),
                        "k_cache": MLFeatureValue(multiArray: state.kCache),
                        "v_cache": MLFeatureValue(multiArray: state.vCache),
                    ])
                }
                output = try GPTSoVITSRuntimeProfiler.measure("t2s.decode.prediction") {
                    try decodeModel.prediction(from: provider)
                }
                if ProcessInfo.processInfo.environment["GPTSOVITS_DEBUG_T2S_RAW"] == "1",
                   state.generatedTokens.count - state.promptTokenCount < 4 {
                    emitRawDebug(
                        prefix: "decode_step_\(state.generatedTokens.count - state.promptTokenCount + 1)",
                        provider: output,
                        generatedTokenCount: state.generatedTokens.count - state.promptTokenCount
                    )
                }
            }
            let logits = try requireArray(named: "logits", from: output)
            let acceptedGeneratedTokenCount = max(state.generatedTokens.count - state.promptTokenCount, 0)
            let resolvedSampling = resolvedSamplingParameters()
            let sampling: T2SSamplingResult
            if resolvedSampling.mode == "coreml_greedy_token",
               prefersModelGreedyToken(),
               let modelSampledToken = sampledToken(from: output) {
                let modelEOS = eosReached(from: output) ?? (modelSampledToken == Int32(manifest.runtime.eosToken))
                sampling = T2SSamplingResult(
                    sampledToken: modelSampledToken,
                    greedyToken: modelSampledToken,
                    shouldStop: modelEOS
                )
            } else {
                let hostSampling = sampleTokenFromLogits(
                    logits,
                    previousTokens: state.generatedTokens,
                    generatedTokenCount: acceptedGeneratedTokenCount,
                    sampling: resolvedSampling,
                    rngState: &state.samplingRNGState,
                    rngBox: state.samplingRNGBox
                )
                sampling = T2SSamplingResult(
                    sampledToken: hostSampling.sampledToken,
                    greedyToken: hostSampling.greedyToken,
                    shouldStop: hostSampling.shouldStop || (eosReached(from: output) ?? false)
                )
            }
            let nextToken = sampling.shouldStop ? Int32(manifest.runtime.eosToken) : sampling.sampledToken
            state.lastToken = try GPTSoVITSRuntimeProfiler.measure("t2s.decode.output.last_token_materialize") {
                try makeInt32Array(shape: [1, 1], values: [nextToken])
            }
            if !sampling.shouldStop {
                state.generatedTokens.append(nextToken)
            }
            state.cacheLen = try GPTSoVITSRuntimeProfiler.measure("t2s.decode.output.cache_len_materialize") {
                try materializeArray(try requireArray(named: "next_cache_len", from: output))
            }
            state.nextPosition = try GPTSoVITSRuntimeProfiler.measure("t2s.decode.output.next_position_materialize") {
                try materializeArray(try requireArray(named: "next_position", from: output))
            }
            state.kCache = try GPTSoVITSRuntimeProfiler.measure("t2s.decode.output.k_cache_materialize") {
                try materializeArray(try requireArray(named: "next_k_cache", from: output))
            }
            state.vCache = try GPTSoVITSRuntimeProfiler.measure("t2s.decode.output.v_cache_materialize") {
                try materializeArray(try requireArray(named: "next_v_cache", from: output))
            }
            return logits
        }
    }

    public func decodeGreedy(state: inout T2SDecodeState, limit: Int? = nil, stopOnEOS: Bool = true) throws -> [Int32] {
        let maxSteps = min(limit ?? manifest.runtime.maxDecodeSteps, manifest.runtime.maxDecodeSteps)
        var tokens: [Int32] = []
        for _ in 0..<maxSteps {
            _ = try decodeStep(state: &state)
            let token = Int32(truncating: state.lastToken[0])
            if stopOnEOS && tokenIsEOS(from: state.lastToken) {
                break
            }
            tokens.append(token)
        }
        return tokens
    }

    private func requireArray(named name: String, from provider: MLFeatureProvider) throws -> MLMultiArray {
        guard let value = provider.featureValue(for: name)?.multiArrayValue else {
            throw NSError(domain: "T2SCoreMLDriver", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Missing MLMultiArray output '\(name)'."
            ])
        }
        return value
    }

    private func requireShape(_ array: MLMultiArray, rank: Int, name: String) throws -> [Int] {
        let shape = array.shape.map { Int(truncating: $0) }
        guard shape.count == rank else {
            throw NSError(domain: "T2SCoreMLDriver", code: 13, userInfo: [
                NSLocalizedDescriptionKey: "Expected \(name) rank \(rank), got shape \(shape)."
            ])
        }
        return shape
    }

    private func tokenIsEOS(from sampledToken: MLMultiArray) -> Bool {
        Int32(truncating: sampledToken[0]) == Int32(manifest.runtime.eosToken)
    }

    private func sampledToken(from provider: MLFeatureProvider) -> Int32? {
        guard let sampledToken = provider.featureValue(for: "sampled_token")?.multiArrayValue,
              sampledToken.count > 0 else {
            return nil
        }
        return Int32(truncating: sampledToken[0])
    }

    private func eosReached(from provider: MLFeatureProvider) -> Bool? {
        guard let eosReached = provider.featureValue(for: "eos_reached")?.multiArrayValue,
              eosReached.count > 0 else {
            return nil
        }
        return Int32(truncating: eosReached[0]) != 0
    }

    private func int32Values(from array: MLMultiArray, activeCount: Int) throws -> [Int32] {
        guard activeCount >= 0, activeCount <= array.count else {
            throw NSError(domain: "T2SCoreMLDriver", code: 14, userInfo: [
                NSLocalizedDescriptionKey: "Requested activeCount \(activeCount) is outside array bounds \(array.count)."
            ])
        }
        return (0..<activeCount).map { Int32(truncating: array[$0]) }
    }

    private func resolveLengthArray(length: Int?, defaultLength: Int, capacity: Int) throws -> MLMultiArray {
        let resolvedLength = length ?? defaultLength
        guard resolvedLength >= 0, resolvedLength <= capacity else {
            throw NSError(domain: "T2SCoreMLDriver", code: 12, userInfo: [
                NSLocalizedDescriptionKey: "Length \(resolvedLength) is outside valid range [0, \(capacity)]."
            ])
        }
        return try makeInt32Array(shape: [1], values: [Int32(resolvedLength)])
    }

    private func argmax(_ array: MLMultiArray) -> Int32 {
        let values = rawDoubleValues(from: array)
        guard let best = values.enumerated().max(by: { $0.element < $1.element }) else {
            return 0
        }
        return Int32(best.offset)
    }

    private func materializeArray(_ source: MLMultiArray) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: source.shape, dataType: source.dataType)
        if ProcessInfo.processInfo.environment["GPTSOVITS_T2S_SAFE_MATERIALIZE"] == "1" {
            for index in 0..<source.count {
                array[index] = source[index]
            }
            return array
        }
        let shape = source.shape.map { Int(truncating: $0) }
        let sourceStrides = source.strides.map { Int(truncating: $0) }
        let targetStrides = array.strides.map { Int(truncating: $0) }
        let byteCount = source.count * bytesPerElement(for: source.dataType)
        if byteCount > 0, sourceStrides == targetStrides {
            memcpy(array.dataPointer, source.dataPointer, byteCount)
            return array
        }
        switch source.dataType {
        case .double:
            copyStridedElements(
                from: source.dataPointer.bindMemory(to: Double.self, capacity: source.count),
                to: array.dataPointer.bindMemory(to: Double.self, capacity: array.count),
                shape: shape,
                sourceStrides: sourceStrides,
                targetStrides: targetStrides
            )
        case .float32:
            copyStridedElements(
                from: source.dataPointer.bindMemory(to: Float32.self, capacity: source.count),
                to: array.dataPointer.bindMemory(to: Float32.self, capacity: array.count),
                shape: shape,
                sourceStrides: sourceStrides,
                targetStrides: targetStrides
            )
        case .float16:
            copyStridedElements(
                from: source.dataPointer.bindMemory(to: UInt16.self, capacity: source.count),
                to: array.dataPointer.bindMemory(to: UInt16.self, capacity: array.count),
                shape: shape,
                sourceStrides: sourceStrides,
                targetStrides: targetStrides
            )
        case .int32:
            copyStridedElements(
                from: source.dataPointer.bindMemory(to: Int32.self, capacity: source.count),
                to: array.dataPointer.bindMemory(to: Int32.self, capacity: array.count),
                shape: shape,
                sourceStrides: sourceStrides,
                targetStrides: targetStrides
            )
        case .int8:
            copyStridedElements(
                from: source.dataPointer.bindMemory(to: Int8.self, capacity: source.count),
                to: array.dataPointer.bindMemory(to: Int8.self, capacity: array.count),
                shape: shape,
                sourceStrides: sourceStrides,
                targetStrides: targetStrides
            )
        @unknown default:
            for index in 0..<source.count {
                array[index] = source[index]
            }
        }
        return array
    }

    private func copyStridedElements<Element>(
        from source: UnsafePointer<Element>,
        to target: UnsafeMutablePointer<Element>,
        shape: [Int],
        sourceStrides: [Int],
        targetStrides: [Int],
        dimension: Int = 0,
        sourceOffset: Int = 0,
        targetOffset: Int = 0
    ) {
        guard !shape.isEmpty else {
            return
        }
        if dimension == shape.count - 1 {
            let count = shape[dimension]
            let sourceStride = sourceStrides[dimension]
            let targetStride = targetStrides[dimension]
            for index in 0..<count {
                target[targetOffset + index * targetStride] = source[sourceOffset + index * sourceStride]
            }
            return
        }
        let count = shape[dimension]
        let sourceStride = sourceStrides[dimension]
        let targetStride = targetStrides[dimension]
        for index in 0..<count {
            copyStridedElements(
                from: source,
                to: target,
                shape: shape,
                sourceStrides: sourceStrides,
                targetStrides: targetStrides,
                dimension: dimension + 1,
                sourceOffset: sourceOffset + index * sourceStride,
                targetOffset: targetOffset + index * targetStride
            )
        }
    }

    private func bytesPerElement(for dataType: MLMultiArrayDataType) -> Int {
        switch dataType {
        case .double:
            return MemoryLayout<Double>.size
        case .float32:
            return MemoryLayout<Float32>.size
        case .float16:
            return MemoryLayout<UInt16>.size
        case .int32:
            return MemoryLayout<Int32>.size
        case .int8:
            return MemoryLayout<Int8>.size
        @unknown default:
            return 0
        }
    }

    private func resolvedSamplingParameters() -> T2SSamplingParameters {
        let defaults = manifest.runtime.samplingDefaults
        let defaultTopK = defaults?.topK ?? 15
        let defaultTopP = defaults?.topP ?? 1.0
        let defaultTemperature = defaults?.temperature ?? 1.0
        let defaultRepetitionPenalty = defaults?.repetitionPenalty ?? 1.35
        if ProcessInfo.processInfo.environment["GPTSOVITS_FORCE_T2S_GREEDY"] == "1" {
            // Keep the default host-side logit shaping, but replace multinomial with
            // deterministic argmax so debug runs stay closer to the production path.
            return T2SSamplingParameters(
                mode: "host_argmax",
                topK: defaultTopK > 0 ? defaultTopK : nil,
                topP: defaultTopP < 1.0 ? defaultTopP : nil,
                temperature: defaultTemperature,
                repetitionPenalty: defaultRepetitionPenalty
            )
        }
        let mode = defaults?.mode ?? "host_multinomial"
        return T2SSamplingParameters(
            mode: mode,
            topK: defaultTopK > 0 ? defaultTopK : nil,
            topP: defaultTopP < 1.0 ? defaultTopP : nil,
            temperature: defaultTemperature,
            repetitionPenalty: defaultRepetitionPenalty
        )
    }

    private func prefersModelGreedyToken() -> Bool {
        ProcessInfo.processInfo.environment["GPTSOVITS_PREFER_MODEL_GREEDY_TOKEN"] == "1"
    }

    private func debugRawStepLimit() -> Int {
        guard let raw = ProcessInfo.processInfo.environment["GPTSOVITS_DEBUG_T2S_RAW_STEPS"],
              let value = Int(raw),
              value > 0 else {
            return 4
        }
        return value
    }

    private func samplingDebugEnabled(for generatedTokenCount: Int) -> Bool {
        guard ProcessInfo.processInfo.environment["GPTSOVITS_DEBUG_T2S_SAMPLING"] == "1" else {
            return false
        }
        return generatedTokenCount < debugRawStepLimit()
    }

    private func hostArgmaxLoopEscapeRepeatThreshold() -> Int {
        16
    }

    private func trailingRepeatCount(of token: Int32, in previousTokens: [Int32]) -> Int {
        var count = 0
        for previousToken in previousTokens.reversed() {
            if previousToken == token {
                count += 1
            } else {
                break
            }
        }
        return count
    }

    private func argmaxExcludingToken(_ logits: [Double], excludedToken: Int32) -> Int32? {
        let excludedIndex = Int(excludedToken)
        var bestIndex: Int?
        var bestValue = -Double.infinity
        for (index, value) in logits.enumerated() {
            if index == excludedIndex || !value.isFinite {
                continue
            }
            if bestIndex == nil || value > bestValue {
                bestIndex = index
                bestValue = value
            }
        }
        guard let bestIndex else {
            return nil
        }
        return Int32(bestIndex)
    }

    private func argmaxExcludingToken(_ logits: [Float], excludedToken: Int32) -> Int32? {
        let excludedIndex = Int(excludedToken)
        var bestIndex: Int?
        var bestValue = -Float.infinity
        for (index, value) in logits.enumerated() {
            if index == excludedIndex || !value.isFinite {
                continue
            }
            if bestIndex == nil || value > bestValue {
                bestIndex = index
                bestValue = value
            }
        }
        guard let bestIndex else {
            return nil
        }
        return Int32(bestIndex)
    }

    private func sampleTokenFromLogits(
        _ logits: MLMultiArray,
        previousTokens: [Int32],
        generatedTokenCount: Int,
        sampling: T2SSamplingParameters,
        rngState: inout UInt64?,
        rngBox: T2SSamplingRNGBox?
    ) -> T2SSamplingResult {
        let samplingDebug = samplingDebugEnabled(for: generatedTokenCount)
        var adjustedLogits = rawFloatValues(from: logits)
        let eosToken = Int(manifest.runtime.eosToken)
        let suppressEOSToken = generatedTokenCount < 11 && eosToken == adjustedLogits.count - 1
        if suppressEOSToken {
            adjustedLogits.removeLast()
        } else if eosToken >= 0, eosToken < adjustedLogits.count {
            adjustedLogits[eosToken] = -.infinity
        }

        let rawGreedyToken = argmax(adjustedLogits)
        let rawEOSScore = (suppressEOSToken || eosToken < 0 || eosToken >= adjustedLogits.count) ?
            -.infinity :
            Double(adjustedLogits[eosToken])
        if sampling.mode == "coreml_greedy_token" {
            if samplingDebugEnabled(for: generatedTokenCount) {
                emitSamplingDebug(
                    generatedTokenCount: generatedTokenCount,
                    mode: sampling.mode,
                    rawGreedyToken: rawGreedyToken,
                    adjustedGreedyToken: rawGreedyToken,
                    sampledToken: rawGreedyToken,
                    rawEOSScore: rawEOSScore,
                    adjustedEOSScore: rawEOSScore,
                    shouldStop: !suppressEOSToken && rawGreedyToken == Int32(manifest.runtime.eosToken),
                    repeatRun: 0,
                    loopEscapeTriggered: false
                )
            }
            return T2SSamplingResult(
                sampledToken: rawGreedyToken,
                greedyToken: rawGreedyToken,
                shouldStop: !suppressEOSToken && rawGreedyToken == Int32(manifest.runtime.eosToken)
            )
        }
        if sampling.repetitionPenalty != 1.0, !previousTokens.isEmpty {
            for token in Set(previousTokens) {
                let index = Int(token)
                guard index >= 0, index < adjustedLogits.count else {
                    continue
                }
                let score = adjustedLogits[index]
                adjustedLogits[index] = score < 0 ? score * Float(sampling.repetitionPenalty) : score / Float(sampling.repetitionPenalty)
            }
        }
        if let topP = sampling.topP {
            applyTopPFilter(to: &adjustedLogits, topP: topP)
        }
        let temperature = max(Float(sampling.temperature), 1e-5)
        if temperature != 1.0 {
            adjustedLogits = adjustedLogits.map { $0 / temperature }
        }
        if let topK = sampling.topK {
            applyTopKFilter(to: &adjustedLogits, topK: topK)
        }
        var adjustedGreedyToken = argmax(adjustedLogits)
        let repeatRun = trailingRepeatCount(of: adjustedGreedyToken, in: previousTokens)
        var loopEscapeTriggered = false
        if sampling.mode == "host_argmax",
           adjustedGreedyToken != Int32(manifest.runtime.eosToken),
           repeatRun >= hostArgmaxLoopEscapeRepeatThreshold(),
           let escapedToken = argmaxExcludingToken(adjustedLogits, excludedToken: adjustedGreedyToken) {
            // Deterministic argmax can get stuck in long self-loops on some cases;
            // when that happens, step to the best alternative token instead.
            adjustedGreedyToken = escapedToken
            loopEscapeTriggered = true
        }
        let adjustedEOSScore = (suppressEOSToken || eosToken < 0 || eosToken >= adjustedLogits.count) ?
            -.infinity :
            Double(adjustedLogits[eosToken])
        if sampling.mode == "host_argmax" {
            let shouldStop = !suppressEOSToken && (
                rawGreedyToken == Int32(manifest.runtime.eosToken) ||
                adjustedGreedyToken == Int32(manifest.runtime.eosToken)
            )
            if samplingDebugEnabled(for: generatedTokenCount) {
                emitSamplingDebug(
                    generatedTokenCount: generatedTokenCount,
                    mode: sampling.mode,
                    rawGreedyToken: rawGreedyToken,
                    adjustedGreedyToken: adjustedGreedyToken,
                    sampledToken: adjustedGreedyToken,
                    rawEOSScore: rawEOSScore,
                    adjustedEOSScore: adjustedEOSScore,
                    shouldStop: shouldStop,
                    repeatRun: repeatRun,
                    loopEscapeTriggered: loopEscapeTriggered
                )
            }
            return T2SSamplingResult(
                sampledToken: adjustedGreedyToken,
                greedyToken: rawGreedyToken,
                shouldStop: shouldStop
            )
        }

        let probabilities = softmax(adjustedLogits)
        let topProbabilityEntries = samplingDebug ? debugTopEntries(probabilities, count: 8) : nil
        if probabilities.allSatisfy({ $0 == 0 || !$0.isFinite }) {
            let shouldStop = !suppressEOSToken && (
                rawGreedyToken == Int32(manifest.runtime.eosToken) ||
                adjustedGreedyToken == Int32(manifest.runtime.eosToken)
            )
            if samplingDebug {
                emitSamplingDebug(
                    generatedTokenCount: generatedTokenCount,
                    mode: sampling.mode,
                    rawGreedyToken: rawGreedyToken,
                    adjustedGreedyToken: adjustedGreedyToken,
                    sampledToken: adjustedGreedyToken,
                    rawEOSScore: rawEOSScore,
                    adjustedEOSScore: adjustedEOSScore,
                    shouldStop: shouldStop,
                    repeatRun: repeatRun,
                    loopEscapeTriggered: loopEscapeTriggered,
                    topProbabilityEntries: topProbabilityEntries
                )
            }
            return T2SSamplingResult(
                sampledToken: adjustedGreedyToken,
                greedyToken: rawGreedyToken,
                shouldStop: shouldStop
            )
        }
        let topScoreEntries: T2SDebugTopEntries?
        let sampledIndex: Int
        if samplingDebug, let rngBox {
            let debugSample = sampleIndexTorchLikeDebug(from: probabilities, rngBox: rngBox, count: 8)
            sampledIndex = debugSample.index
            topScoreEntries = debugSample.topScoreEntries
        } else {
            sampledIndex = sampleIndex(
                from: probabilities,
                rngState: &rngState,
                rngBox: rngBox
            )
            topScoreEntries = nil
        }
        let sampledToken = Int32(sampledIndex)
        let shouldStop = !suppressEOSToken && (
            rawGreedyToken == Int32(manifest.runtime.eosToken) ||
            sampledToken == Int32(manifest.runtime.eosToken)
        )
        if samplingDebug {
            emitSamplingDebug(
                generatedTokenCount: generatedTokenCount,
                mode: sampling.mode,
                rawGreedyToken: rawGreedyToken,
                adjustedGreedyToken: adjustedGreedyToken,
                sampledToken: sampledToken,
                rawEOSScore: rawEOSScore,
                adjustedEOSScore: adjustedEOSScore,
                shouldStop: shouldStop,
                repeatRun: repeatRun,
                loopEscapeTriggered: loopEscapeTriggered,
                topProbabilityEntries: topProbabilityEntries,
                topScoreEntries: topScoreEntries
            )
        }
        return T2SSamplingResult(
            sampledToken: sampledToken,
            greedyToken: rawGreedyToken,
            shouldStop: shouldStop
        )
    }

    private func argmax(_ logits: [Double]) -> Int32 {
        guard let best = logits.enumerated().max(by: { $0.element < $1.element }) else {
            return 0
        }
        return Int32(best.offset)
    }

    private func argmax(_ logits: [Float]) -> Int32 {
        guard let best = logits.enumerated().max(by: { $0.element < $1.element }) else {
            return 0
        }
        return Int32(best.offset)
    }

    private func applyTopKFilter(to logits: inout [Double], topK: Int) {
        guard topK > 0, topK < logits.count else {
            return
        }
        let sorted = logits.enumerated().sorted { $0.element > $1.element }
        let threshold = sorted[topK - 1].element
        for index in logits.indices where logits[index] < threshold {
            logits[index] = -.infinity
        }
    }

    private func applyTopKFilter(to logits: inout [Float], topK: Int) {
        guard topK > 0, topK < logits.count else {
            return
        }
        let sorted = logits.enumerated().sorted { $0.element > $1.element }
        let threshold = sorted[topK - 1].element
        for index in logits.indices where logits[index] < threshold {
            logits[index] = -.infinity
        }
    }

    private func applyTopPFilter(to logits: inout [Double], topP: Double) {
        guard topP < 1.0 else {
            return
        }
        let sorted = logits.enumerated().sorted { $0.element > $1.element }
        let sortedLogits = sorted.map(\.element)
        let sortedProbabilities = softmax(sortedLogits)
        var cumulativeProbability = 0.0
        for (sortedIndex, item) in sorted.enumerated() {
            cumulativeProbability += sortedProbabilities[sortedIndex]
            if sortedIndex > 0, cumulativeProbability > topP {
                logits[item.offset] = -.infinity
            }
        }
    }

    private func applyTopPFilter(to logits: inout [Float], topP: Double) {
        guard topP < 1.0 else {
            return
        }
        let sorted = logits.enumerated().sorted { $0.element > $1.element }
        let sortedLogits = sorted.map(\.element)
        let sortedProbabilities = softmax(sortedLogits)
        var cumulativeProbability: Float = 0
        let threshold = Float(topP)
        for (sortedIndex, item) in sorted.enumerated() {
            cumulativeProbability += sortedProbabilities[sortedIndex]
            if sortedIndex > 0, cumulativeProbability > threshold {
                logits[item.offset] = -.infinity
            }
        }
    }

    private func softmax(_ logits: [Double]) -> [Double] {
        let finiteLogits = logits.filter { $0.isFinite }
        guard let maxLogit = finiteLogits.max() else {
            return Array(repeating: 0, count: logits.count)
        }
        let exponentials = logits.map { value -> Double in
            guard value.isFinite else {
                return 0
            }
            return Foundation.exp(value - maxLogit)
        }
        let total = exponentials.reduce(0, +)
        guard total > 0, total.isFinite else {
            return Array(repeating: 0, count: logits.count)
        }
        return exponentials.map { $0 / total }
    }

    private func softmax(_ logits: [Float]) -> [Float] {
        let finiteLogits = logits.filter { $0.isFinite }
        guard let maxLogit = finiteLogits.max() else {
            return Array(repeating: 0, count: logits.count)
        }
        let exponentials = logits.map { value -> Float in
            guard value.isFinite else {
                return 0
            }
            return Float(Foundation.exp(Double(value - maxLogit)))
        }
        let total = exponentials.reduce(0, +)
        guard total > 0, total.isFinite else {
            return Array(repeating: 0, count: logits.count)
        }
        return exponentials.map { $0 / total }
    }

    private func sampleIndex(
        from probabilities: [Double],
        rngState: inout UInt64?,
        rngBox: T2SSamplingRNGBox?
    ) -> Int {
        if let rngBox {
            return sampleIndexTorchLike(from: probabilities, rngBox: rngBox)
        }
        let randomValue: Double
        if let state = rngState {
            var rng = SplitMix64(seed: state)
            randomValue = uniformRandom(in: 0.0..<1.0, using: &rng)
            rngState = rng.state
        } else {
            var rng = SystemRandomNumberGenerator()
            randomValue = uniformRandom(in: 0.0..<1.0, using: &rng)
        }

        var cumulativeProbability = 0.0
        for (index, probability) in probabilities.enumerated() {
            cumulativeProbability += probability
            if randomValue <= cumulativeProbability {
                return index
            }
        }
        return probabilities.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
    }

    private func sampleIndex(
        from probabilities: [Float],
        rngState: inout UInt64?,
        rngBox: T2SSamplingRNGBox?
    ) -> Int {
        if let rngBox {
            return sampleIndexTorchLike(from: probabilities, rngBox: rngBox)
        }
        let randomValue: Float
        if let state = rngState {
            var rng = SplitMix64(seed: state)
            randomValue = Float(uniformRandom(in: 0.0..<1.0, using: &rng))
            rngState = rng.state
        } else {
            var rng = SystemRandomNumberGenerator()
            randomValue = Float(uniformRandom(in: 0.0..<1.0, using: &rng))
        }

        var cumulativeProbability: Float = 0
        for (index, probability) in probabilities.enumerated() {
            cumulativeProbability += probability
            if randomValue <= cumulativeProbability {
                return index
            }
        }
        return probabilities.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
    }

    private func sampleIndexTorchLike(from probabilities: [Double], rngBox: T2SSamplingRNGBox) -> Int {
        var bestIndex = 0
        var bestScore = -Double.infinity
        for (index, probability) in probabilities.enumerated() {
            let unit = rngBox.nextUnitDouble53()
            let exponential = -Foundation.log1p(-unit)
            guard probability > 0, probability.isFinite else {
                continue
            }
            let score = probability / exponential
            if score > bestScore {
                bestScore = score
                bestIndex = index
            }
        }
        return bestScore.isFinite ? bestIndex : (probabilities.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0)
    }

    private func sampleIndexTorchLike(from probabilities: [Float], rngBox: T2SSamplingRNGBox) -> Int {
        var bestIndex = 0
        var bestScore = -Float.infinity
        for (index, probability) in probabilities.enumerated() {
            let unit = rngBox.nextUnitDouble53()
            let exponential = Float(-Foundation.log1p(-unit))
            guard probability > 0, probability.isFinite else {
                continue
            }
            let score = probability / exponential
            if score > bestScore {
                bestScore = score
                bestIndex = index
            }
        }
        return bestScore.isFinite ? bestIndex : (probabilities.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0)
    }

    private func sampleIndexTorchLikeDebug(
        from probabilities: [Float],
        rngBox: T2SSamplingRNGBox,
        count: Int
    ) -> (index: Int, topScoreEntries: T2SDebugTopEntries) {
        var bestIndex = 0
        var bestScore = -Float.infinity
        var scores = Array(repeating: -Float.infinity, count: probabilities.count)
        for (index, probability) in probabilities.enumerated() {
            let unit = rngBox.nextUnitDouble53()
            let exponential = Float(-Foundation.log1p(-unit))
            guard probability > 0, probability.isFinite else {
                continue
            }
            let score = probability / exponential
            scores[index] = score
            if score > bestScore {
                bestScore = score
                bestIndex = index
            }
        }
        let topScoreEntries = debugTopEntries(scores, count: count)
        let resolvedIndex = bestScore.isFinite ? bestIndex : (probabilities.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0)
        return (resolvedIndex, topScoreEntries)
    }

    private func uniformRandom<R: RandomNumberGenerator>(in range: Range<Double>, using rng: inout R) -> Double {
        let value = Double(rng.next()) / Double(UInt64.max)
        return range.lowerBound + (range.upperBound - range.lowerBound) * value
    }

    private func rawDoubleValues(from array: MLMultiArray) -> [Double] {
        (0..<array.count).map { rawOffset in
            rawDoubleValue(in: array, rawOffset: rawOffset)
        }
    }

    private func rawFloatValues(from array: MLMultiArray) -> [Float] {
        (0..<array.count).map { rawOffset in
            Float(rawDoubleValue(in: array, rawOffset: rawOffset))
        }
    }

    private func debugTopEntries(_ values: [Float], count: Int) -> T2SDebugTopEntries {
        let pairs = values.enumerated().filter { $0.element.isFinite }.sorted { $0.element > $1.element }.prefix(max(count, 0))
        return T2SDebugTopEntries(
            tokens: pairs.map(\.offset),
            values: pairs.map(\.element)
        )
    }

    private func rawInt32Value(in array: MLMultiArray, rawOffset: Int = 0) -> Int32 {
        if ProcessInfo.processInfo.environment["GPTSOVITS_T2S_SAFE_READ"] == "1" {
            return Int32(truncating: array[rawOffset])
        }
        switch array.dataType {
        case .int32:
            return array.dataPointer.bindMemory(to: Int32.self, capacity: rawOffset + 1)[rawOffset]
        case .int8:
            return Int32(array.dataPointer.bindMemory(to: Int8.self, capacity: rawOffset + 1)[rawOffset])
        case .float32:
            return Int32(array.dataPointer.bindMemory(to: Float32.self, capacity: rawOffset + 1)[rawOffset])
        case .double:
            return Int32(array.dataPointer.bindMemory(to: Double.self, capacity: rawOffset + 1)[rawOffset])
        case .float16:
            let bits = array.dataPointer.bindMemory(to: UInt16.self, capacity: rawOffset + 1)[rawOffset]
            return Int32(Float(Float16(bitPattern: bits)))
        @unknown default:
            return Int32(truncating: array[rawOffset])
        }
    }

    private func rawDoubleValue(in array: MLMultiArray, rawOffset: Int) -> Double {
        if ProcessInfo.processInfo.environment["GPTSOVITS_T2S_SAFE_READ"] == "1" {
            return array[rawOffset].doubleValue
        }
        switch array.dataType {
        case .double:
            return array.dataPointer.bindMemory(to: Double.self, capacity: rawOffset + 1)[rawOffset]
        case .float32:
            return Double(array.dataPointer.bindMemory(to: Float32.self, capacity: rawOffset + 1)[rawOffset])
        case .float16:
            let bits = array.dataPointer.bindMemory(to: UInt16.self, capacity: rawOffset + 1)[rawOffset]
            return Double(Float(Float16(bitPattern: bits)))
        case .int32:
            return Double(array.dataPointer.bindMemory(to: Int32.self, capacity: rawOffset + 1)[rawOffset])
        case .int8:
            return Double(array.dataPointer.bindMemory(to: Int8.self, capacity: rawOffset + 1)[rawOffset])
        @unknown default:
            return array[rawOffset].doubleValue
        }
    }

    private static func loadModel(at url: URL, configuration: MLModelConfiguration) throws -> MLModel {
        try GPTSoVITSCoreMLModelLoader.loadModel(at: url, configuration: configuration)
    }

    private func emitRawDebug(prefix: String, provider: MLFeatureProvider, generatedTokenCount: Int) {
        guard let logits = provider.featureValue(for: "logits")?.multiArrayValue else {
            fputs("[T2S RAW] \(prefix) missing logits\n", stderr)
            return
        }
        let rawLogits = rawDoubleValues(from: logits)
        let eosToken = Int(manifest.runtime.eosToken)
        let topPairs = rawLogits.enumerated().sorted { $0.element > $1.element }.prefix(8)
        let topTokens = topPairs.map(\.offset)
        let topValues = topPairs.map { $0.element }
        let sampledTokenValue: Int32? = provider.featureValue(for: "sampled_token")?.multiArrayValue.map { rawInt32Value(in: $0) }
        let eosReachedValue: Int32? = provider.featureValue(for: "eos_reached")?.multiArrayValue.map { rawInt32Value(in: $0) }
        let eosLogit = (eosToken >= 0 && eosToken < rawLogits.count) ? rawLogits[eosToken] : .nan
        let line = "[T2S RAW] \(prefix) generated=\(generatedTokenCount) shape=\(logits.shape.map { Int(truncating: $0) }) sampled=\(sampledTokenValue.map(String.init) ?? "nil") eos=\(eosReachedValue.map(String.init) ?? "nil") eos_logit=\(eosLogit) top_tokens=\(topTokens) top_values=\(topValues)\n"
        fputs(line, stderr)
    }

    private func emitPrepareDebug(prefix: String, provider: MLFeatureProvider) {
        var parts = ["[T2S PREPARE] \(prefix)"]
        if let xyPos = provider.featureValue(for: "xy_pos")?.multiArrayValue {
            let values = rawDoubleValues(from: xyPos)
            let finiteValues = values.filter(\.isFinite)
            let nanCount = values.count - finiteValues.count
            parts.append("xy_pos_shape=\(xyPos.shape.map { Int(truncating: $0) })")
            parts.append("xy_pos_nan=\(nanCount)")
            if let minValue = finiteValues.min(), let maxValue = finiteValues.max() {
                parts.append("xy_pos_min=\(minValue)")
                parts.append("xy_pos_max=\(maxValue)")
            }
        }
        if let promptAttnMask = provider.featureValue(for: "prompt_attn_mask")?.multiArrayValue {
            let uniqueValues = Array(Set((0..<promptAttnMask.count).map { rawInt32Value(in: promptAttnMask, rawOffset: $0) })).sorted()
            parts.append("prompt_attn_mask_shape=\(promptAttnMask.shape.map { Int(truncating: $0) })")
            parts.append("prompt_attn_mask_unique=\(Array(uniqueValues.prefix(8)))")
        }
        if let activeSrcLen = provider.featureValue(for: "active_src_len")?.multiArrayValue {
            parts.append("active_src_len=\(rawInt32Value(in: activeSrcLen))")
        }
        if let promptLen = provider.featureValue(for: "prompt_len")?.multiArrayValue {
            parts.append("prompt_len=\(rawInt32Value(in: promptLen))")
        }
        if let nextPosition = provider.featureValue(for: "next_position")?.multiArrayValue {
            parts.append("next_position=\(rawInt32Value(in: nextPosition))")
        }
        let line = parts.joined(separator: " ") + "\n"
        fputs(line, stderr)
    }

    private func emitInputDebug(prefix: String, arrays: [String: MLMultiArray]) {
        for (name, array) in arrays.sorted(by: { $0.key < $1.key }) {
            let shape = array.shape.map { Int(truncating: $0) }
            let strides = array.strides.map { Int(truncating: $0) }
            var line = "[T2S INPUT] \(prefix) \(name) shape=\(shape) strides=\(strides) dtype=\(array.dataType.rawValue)"
            switch array.dataType {
            case .float32, .float16, .double:
                let values = rawDoubleValues(from: array)
                let finiteValues = values.filter(\.isFinite)
                let nanCount = values.count - finiteValues.count
                line += " nan=\(nanCount)"
                if let minValue = finiteValues.min(), let maxValue = finiteValues.max() {
                    line += " min=\(minValue) max=\(maxValue)"
                }
            default:
                let preview = (0..<min(array.count, 8)).map { rawInt32Value(in: array, rawOffset: $0) }
                line += " preview=\(preview)"
            }
            line += "\n"
            fputs(line, stderr)
        }
    }

    private func emitSamplingDebug(
        generatedTokenCount: Int,
        mode: String,
        rawGreedyToken: Int32,
        adjustedGreedyToken: Int32,
        sampledToken: Int32,
        rawEOSScore: Double,
        adjustedEOSScore: Double,
        shouldStop: Bool,
        repeatRun: Int,
        loopEscapeTriggered: Bool,
        topProbabilityEntries: T2SDebugTopEntries? = nil,
        topScoreEntries: T2SDebugTopEntries? = nil
    ) {
        var line = "[T2S SAMPLE] generated=\(generatedTokenCount) mode=\(mode) raw_greedy=\(rawGreedyToken) adjusted_greedy=\(adjustedGreedyToken) sampled=\(sampledToken) raw_eos_score=\(rawEOSScore) adjusted_eos_score=\(adjustedEOSScore) repeat_run=\(repeatRun) loop_escape=\(loopEscapeTriggered) stop=\(shouldStop)"
        if let topProbabilityEntries {
            line += " top_prob_tokens=\(topProbabilityEntries.tokens) top_prob_values=\(topProbabilityEntries.values)"
        }
        if let topScoreEntries {
            line += " top_score_tokens=\(topScoreEntries.tokens) top_score_values=\(topScoreEntries.values)"
        }
        line += "\n"
        fputs(line, stderr)
    }
}
