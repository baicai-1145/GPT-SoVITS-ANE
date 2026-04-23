import CoreML
import Foundation

public struct VITSBundleManifest: Decodable {
    public struct Artifact: Decodable {
        public let target: String
        public let filename: String
        public let path: String
    }

    public struct Artifacts: Decodable {
        public let decodeCondition: Artifact
        public let prior: Artifact
        public let latentSampler: Artifact?
        public let flow: Artifact
        public let waveGenerator: Artifact
        public let speakerEncoder: Artifact?

        private enum CodingKeys: String, CodingKey {
            case decodeCondition = "decode_condition"
            case prior
            case latentSampler = "latent_sampler"
            case flow
            case waveGenerator = "wave_generator"
            case speakerEncoder = "speaker_encoder"
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

            public let referFrameLen: Int
            public let referFrameCountRange: ShapeRange?
            public let semanticCodeLen: Int
            public let textPhoneLen: Int
            public let latentLen: Int

            private enum CodingKeys: String, CodingKey {
                case referFrameLen = "refer_frame_len"
                case referFrameCountRange = "refer_frame_count_range"
                case semanticCodeLen = "semantic_code_len"
                case textPhoneLen = "text_phone_len"
                case latentLen = "latent_len"
            }
        }

        public struct ReferenceAudioContract: Decodable {
            public struct ReferModelInputContract: Decodable {
                public struct ShapeRange: Decodable {
                    public let lowerBound: Int
                    public let upperBound: Int

                    private enum CodingKeys: String, CodingKey {
                        case lowerBound = "lower_bound"
                        case upperBound = "upper_bound"
                    }
                }

                public let source: String
                public let channelSliceStart: Int
                public let channelSliceEnd: Int
                public let targetFrameLength: Int
                public let frameCountRange: ShapeRange?
                public let framePadding: String

                private enum CodingKeys: String, CodingKey {
                    case source
                    case channelSliceStart = "channel_slice_start"
                    case channelSliceEnd = "channel_slice_end"
                    case targetFrameLength = "target_frame_length"
                    case frameCountRange = "frame_count_range"
                    case framePadding = "frame_padding"
                }
            }

            public struct SharedWaveformContract: Decodable {
                public struct AmplitudeNormalization: Decodable {
                    public let condition: String
                    public let formula: String
                }

                public let channelPolicy: String
                public let amplitudeNormalization: AmplitudeNormalization

                private enum CodingKeys: String, CodingKey {
                    case channelPolicy = "channel_policy"
                    case amplitudeNormalization = "amplitude_normalization"
                }
            }

            public struct ReferSpectrogramContract: Decodable {
                public let targetSampleRate: Int
                public let spectrogramType: String
                public let nFFT: Int
                public let hopLength: Int
                public let winLength: Int
                public let center: Bool
                public let padMode: String
                public let onesided: Bool
                public let magnitudeEpsilon: Double
                public let expectedFrequencyBins: Int

                private enum CodingKeys: String, CodingKey {
                    case targetSampleRate = "target_sample_rate"
                    case spectrogramType = "spectrogram_type"
                    case nFFT = "n_fft"
                    case hopLength = "hop_length"
                    case winLength = "win_length"
                    case center
                    case padMode = "pad_mode"
                    case onesided
                    case magnitudeEpsilon = "magnitude_epsilon"
                    case expectedFrequencyBins = "expected_frequency_bins"
                }
            }

            public struct SpeakerFbankContract: Decodable {
                public struct ShapeRange: Decodable {
                    public let lowerBound: Int
                    public let upperBound: Int

                    private enum CodingKeys: String, CodingKey {
                        case lowerBound = "lower_bound"
                        case upperBound = "upper_bound"
                    }
                }

                public let source: String
                public let targetSampleRate: Int
                public let featureType: String
                public let numMelBins: Int
                public let dither: Double
                public let frameCountRange: ShapeRange?
                public let exampleInputShape: [Int]

                private enum CodingKeys: String, CodingKey {
                    case source
                    case targetSampleRate = "target_sample_rate"
                    case featureType = "feature_type"
                    case numMelBins = "num_mel_bins"
                    case dither
                    case frameCountRange = "frame_count_range"
                    case exampleInputShape = "example_input_shape"
                }
            }

            public let sharedWaveformContract: SharedWaveformContract
            public let referSpectrogramContract: ReferSpectrogramContract
            public let referModelInputContract: ReferModelInputContract?
            public let speakerFbank80Contract: SpeakerFbankContract

            private enum CodingKeys: String, CodingKey {
                case sharedWaveformContract = "shared_waveform_contract"
                case referSpectrogramContract = "refer_spectrogram_contract"
                case referModelInputContract = "refer_model_input_contract"
                case speakerFbank80Contract = "speaker_fbank_80_contract"
            }
        }

        public struct DriverContract: Decodable {
            public struct SamplingContract: Decodable {
                public let location: String
                public let formula: String
                public let noiseDistribution: String
                public let noiseScale: Float

                private enum CodingKeys: String, CodingKey {
                    case location
                    case formula
                    case noiseDistribution = "noise_distribution"
                    case noiseScale = "noise_scale"
                }
            }

            public struct PostFlowContract: Decodable {
                public let location: String
                public let formula: String

                private enum CodingKeys: String, CodingKey {
                    case location
                    case formula
                }
            }

            public let driver: String
            public let samplingContract: SamplingContract
            public let postFlowContract: PostFlowContract

            private enum CodingKeys: String, CodingKey {
                case driver
                case samplingContract = "sampling_contract"
                case postFlowContract = "post_flow_contract"
            }
        }

        public let shapes: Shapes
        public let driverContract: DriverContract
        public let referenceAudioContract: ReferenceAudioContract?

        private enum CodingKeys: String, CodingKey {
            case shapes
            case driverContract = "driver_contract"
            case referenceAudioContract = "reference_audio_contract"
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

public struct VITSConditionState {
    public let ge: MLMultiArray
    public let geText: MLMultiArray
}

public struct VITSPriorState {
    public let priorMean: MLMultiArray
    public let priorLogScale: MLMultiArray
    public let yMask: MLMultiArray
}

public struct VITSLatentState {
    public let zP: MLMultiArray
    public let z: MLMultiArray
    public let maskedZ: MLMultiArray
}

public struct VITSDecodeResult {
    public let condition: VITSConditionState
    public let prior: VITSPriorState
    public let latent: VITSLatentState
    public let audio: MLMultiArray
}

private struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64

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

public final class VITSCoreMLDriver {
    public let manifest: VITSBundleManifest
    public let decodeConditionModel: MLModel
    public let priorModel: MLModel
    public let latentSamplerModel: MLModel?
    public let flowModel: MLModel
    public let waveGeneratorModel: MLModel

    public init(bundleDirectory: URL, configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        let manifestURL = bundleDirectory.appendingPathComponent("manifest.json")
        let manifestData = try GPTSoVITSRuntimeProfiler.measure("vits.init.manifest.read") {
            try Data(contentsOf: manifestURL)
        }
        let decoder = JSONDecoder()
        self.manifest = try GPTSoVITSRuntimeProfiler.measure("vits.init.manifest.decode") {
            try decoder.decode(VITSBundleManifest.self, from: manifestData)
        }

        let decodeConditionURL = bundleDirectory.appendingPathComponent(manifest.artifacts.decodeCondition.filename)
        let priorURL = bundleDirectory.appendingPathComponent(manifest.artifacts.prior.filename)
        let latentSamplerURL = manifest.artifacts.latentSampler.map {
            bundleDirectory.appendingPathComponent($0.filename)
        }
        let flowURL = bundleDirectory.appendingPathComponent(manifest.artifacts.flow.filename)
        let waveGeneratorURL = bundleDirectory.appendingPathComponent(manifest.artifacts.waveGenerator.filename)
        self.decodeConditionModel = try GPTSoVITSRuntimeProfiler.measure("vits.init.decode_condition.load") {
            try Self.loadModel(at: decodeConditionURL, configuration: configuration)
        }
        self.priorModel = try GPTSoVITSRuntimeProfiler.measure("vits.init.prior.load") {
            try Self.loadModel(at: priorURL, configuration: configuration)
        }
        self.latentSamplerModel = try GPTSoVITSRuntimeProfiler.measure("vits.init.latent_sampler.load") {
            try latentSamplerURL.map { try Self.loadModel(at: $0, configuration: configuration) }
        }
        self.flowModel = try GPTSoVITSRuntimeProfiler.measure("vits.init.flow.load") {
            try Self.loadModel(at: flowURL, configuration: configuration)
        }
        self.waveGeneratorModel = try GPTSoVITSRuntimeProfiler.measure("vits.init.wave_generator.load") {
            try Self.loadModel(at: waveGeneratorURL, configuration: configuration)
        }
    }

    public func makeInt32Array(shape: [Int], values: [Int32]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map(NSNumber.init(value:)), dataType: .int32)
        guard array.count == values.count else {
            throw NSError(domain: "VITSCoreMLDriver", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Value count \(values.count) does not match shape element count \(array.count)."
            ])
        }
        for (index, value) in values.enumerated() {
            array[index] = NSNumber(value: value)
        }
        return array
    }

    public func makeFloat32Array(shape: [Int], values: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map(NSNumber.init(value:)), dataType: .float32)
        guard array.count == values.count else {
            throw NSError(domain: "VITSCoreMLDriver", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Value count \(values.count) does not match shape element count \(array.count)."
            ])
        }
        for (index, value) in values.enumerated() {
            array[index] = NSNumber(value: value)
        }
        return array
    }

    public func makeLengthArray(values: [Int32]) throws -> MLMultiArray {
        try makeInt32Array(shape: [values.count], values: values)
    }

    public func makeZeroFloat32Array(shape: [Int]) throws -> MLMultiArray {
        return try MLMultiArray(shape: shape.map(NSNumber.init(value:)), dataType: .float32)
    }

    public func decodeCondition(refer: MLMultiArray, svEmb: MLMultiArray) throws -> VITSConditionState {
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "refer": MLFeatureValue(multiArray: refer),
            "sv_emb": MLFeatureValue(multiArray: svEmb),
        ])
        let output = try decodeConditionModel.prediction(from: provider)
        return VITSConditionState(
            ge: try requireArray(named: "ge", from: output),
            geText: try requireArray(named: "ge_text", from: output)
        )
    }

    public func prior(
        codes: MLMultiArray,
        text: MLMultiArray,
        geText: MLMultiArray,
        codeLengths: [Int32]? = nil,
        textLengths: [Int32]? = nil
    ) throws -> VITSPriorState {
        let codeShape = try requireShape(codes, rank: 3, name: "codes")
        let textShape = try requireShape(text, rank: 2, name: "text")
        guard codeShape[1] == textShape[0] else {
            throw NSError(domain: "VITSCoreMLDriver", code: 7, userInfo: [
                NSLocalizedDescriptionKey: "codes batch size \(codeShape[1]) does not match text batch size \(textShape[0])."
            ])
        }

        var dictionary: [String: MLFeatureValue] = [
            "codes": MLFeatureValue(multiArray: codes),
            "text": MLFeatureValue(multiArray: text),
            "ge_text": MLFeatureValue(multiArray: geText),
        ]
        if priorModel.modelDescription.inputDescriptionsByName["code_lengths"] != nil {
            dictionary["code_lengths"] = MLFeatureValue(
                multiArray: try resolveLengthArray(
                    name: "code_lengths",
                    values: codeLengths,
                    batchSize: codeShape[1],
                    capacity: codeShape[2]
                )
            )
        }
        if priorModel.modelDescription.inputDescriptionsByName["text_lengths"] != nil {
            dictionary["text_lengths"] = MLFeatureValue(
                multiArray: try resolveLengthArray(
                    name: "text_lengths",
                    values: textLengths,
                    batchSize: textShape[0],
                    capacity: textShape[1]
                )
            )
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: dictionary)
        let output = try priorModel.prediction(from: provider)
        return VITSPriorState(
            priorMean: try requireArray(named: "prior_mean", from: output),
            priorLogScale: try requireArray(named: "prior_log_scale", from: output),
            yMask: try requireArray(named: "y_mask", from: output)
        )
    }

    public func sampleLatent(
        priorMean: MLMultiArray,
        priorLogScale: MLMultiArray,
        noise: MLMultiArray? = nil,
        noiseScale: Float? = nil,
        seed: UInt64? = nil
    ) throws -> MLMultiArray {
        let resolvedNoiseScale = noiseScale ?? manifest.runtime.driverContract.samplingContract.noiseScale
        if let latentSamplerModel {
            let shape = try requireShape(priorMean, rank: 3, name: "prior_mean")
            let resolvedNoise = try resolveNoiseArray(
                explicitNoise: noise,
                expectedShape: shape,
                seed: seed
            )
            let noiseScaleArray = try makeFloat32Array(shape: [1], values: [resolvedNoiseScale])
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "prior_mean": MLFeatureValue(multiArray: priorMean),
                "prior_log_scale": MLFeatureValue(multiArray: priorLogScale),
                "noise": MLFeatureValue(multiArray: resolvedNoise),
                "noise_scale": MLFeatureValue(multiArray: noiseScaleArray),
            ])
            let output = try latentSamplerModel.prediction(from: provider)
            return try requireArray(named: "z_p", from: output)
        }
        if let explicitNoise = noise {
            return try sampleLatentHost(
                priorMean: priorMean,
                priorLogScale: priorLogScale,
                explicitNoise: explicitNoise,
                noiseScale: resolvedNoiseScale
            )
        }
        if let seed {
            var rng = SplitMix64(seed: seed)
            return try sampleLatentHost(
                priorMean: priorMean,
                priorLogScale: priorLogScale,
                noiseScale: resolvedNoiseScale,
                rng: &rng
            )
        } else {
            var rng = SystemRandomNumberGenerator()
            return try sampleLatentHost(
                priorMean: priorMean,
                priorLogScale: priorLogScale,
                noiseScale: resolvedNoiseScale,
                rng: &rng
            )
        }
    }

    public func flow(zP: MLMultiArray, yMask: MLMultiArray, ge: MLMultiArray) throws -> MLMultiArray {
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "z_p": MLFeatureValue(multiArray: zP),
            "y_mask": MLFeatureValue(multiArray: yMask),
            "ge": MLFeatureValue(multiArray: ge),
        ])
        let output = try flowModel.prediction(from: provider)
        return try requireArray(named: "z", from: output)
    }

    public func maskLatent(_ z: MLMultiArray, yMask: MLMultiArray) throws -> MLMultiArray {
        let zShape = try requireShape(z, rank: 3, name: "z")
        let maskShape = try requireShape(yMask, rank: 3, name: "y_mask")
        guard zShape[0] == maskShape[0], zShape[2] == maskShape[2], maskShape[1] == 1 else {
            throw NSError(domain: "VITSCoreMLDriver", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "Expected y_mask shape [batch, 1, frames] compatible with z shape [batch, channels, frames]."
            ])
        }
        let result = try MLMultiArray(shape: z.shape, dataType: .float32)
        let zStrides = strides(of: z)
        let maskStrides = strides(of: yMask)
        let resultStrides = strides(of: result)
        for batchIndex in 0..<zShape[0] {
            for channelIndex in 0..<zShape[1] {
                for frameIndex in 0..<zShape[2] {
                    let zOffset = offset3D(batch: batchIndex, channel: channelIndex, frame: frameIndex, strides: zStrides)
                    let maskOffset = offset3D(batch: batchIndex, channel: 0, frame: frameIndex, strides: maskStrides)
                    let resultOffset = offset3D(batch: batchIndex, channel: channelIndex, frame: frameIndex, strides: resultStrides)
                    let maskedValue = floatValue(z, at: zOffset) * floatValue(yMask, at: maskOffset)
                    result[resultOffset] = NSNumber(value: maskedValue)
                }
            }
        }
        return result
    }

    public func waveform(z: MLMultiArray, ge: MLMultiArray) throws -> MLMultiArray {
        if waveGeneratorModel.modelDescription.inputDescriptionsByName["y_mask"] != nil {
            let shape = try requireShape(z, rank: 3, name: "z")
            let ones = try MLMultiArray(
                shape: [shape[0], 1, shape[2]].map(NSNumber.init(value:)),
                dataType: .float32
            )
            for index in 0..<ones.count {
                ones[index] = NSNumber(value: 1.0 as Float)
            }
            return try maskedWaveform(z: z, yMask: ones, ge: ge).audio
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "z": MLFeatureValue(multiArray: z),
            "ge": MLFeatureValue(multiArray: ge),
        ])
        let output = try waveGeneratorModel.prediction(from: provider)
        return try requireArray(named: "audio", from: output)
    }

    public func maskedWaveform(
        z: MLMultiArray,
        yMask: MLMultiArray,
        ge: MLMultiArray
    ) throws -> (maskedZ: MLMultiArray, audio: MLMultiArray) {
        if waveGeneratorModel.modelDescription.inputDescriptionsByName["y_mask"] == nil {
            let maskedZ = try maskLatent(z, yMask: yMask)
            let audio = try waveform(z: maskedZ, ge: ge)
            return (maskedZ, audio)
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "z": MLFeatureValue(multiArray: z),
            "y_mask": MLFeatureValue(multiArray: yMask),
            "ge": MLFeatureValue(multiArray: ge),
        ])
        let output = try waveGeneratorModel.prediction(from: provider)
        return (
            maskedZ: try requireArray(named: "masked_z", from: output),
            audio: try requireArray(named: "audio", from: output)
        )
    }

    public func decode(
        codes: MLMultiArray,
        text: MLMultiArray,
        refer: MLMultiArray,
        svEmb: MLMultiArray,
        noise: MLMultiArray? = nil,
        noiseScale: Float? = nil,
        seed: UInt64? = nil,
        codeLengths: [Int32]? = nil,
        textLengths: [Int32]? = nil
    ) throws -> VITSDecodeResult {
        let condition = try decodeCondition(refer: refer, svEmb: svEmb)
        let priorState = try prior(
            codes: codes,
            text: text,
            geText: condition.geText,
            codeLengths: codeLengths,
            textLengths: textLengths
        )
        let zP = try sampleLatent(
            priorMean: priorState.priorMean,
            priorLogScale: priorState.priorLogScale,
            noise: noise,
            noiseScale: noiseScale,
            seed: seed
        )
        let z = try flow(zP: zP, yMask: priorState.yMask, ge: condition.ge)
        let waveformState = try maskedWaveform(z: z, yMask: priorState.yMask, ge: condition.ge)
        return VITSDecodeResult(
            condition: condition,
            prior: priorState,
            latent: VITSLatentState(zP: zP, z: z, maskedZ: waveformState.maskedZ),
            audio: waveformState.audio
        )
    }

    private func requireArray(named name: String, from provider: MLFeatureProvider) throws -> MLMultiArray {
        guard let value = provider.featureValue(for: name)?.multiArrayValue else {
            throw NSError(domain: "VITSCoreMLDriver", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "Missing MLMultiArray output '\(name)'."
            ])
        }
        return value
    }

    private func requireShape(_ array: MLMultiArray, rank: Int, name: String) throws -> [Int] {
        let shape = array.shape.map { Int(truncating: $0) }
        guard shape.count == rank else {
            throw NSError(domain: "VITSCoreMLDriver", code: 5, userInfo: [
                NSLocalizedDescriptionKey: "Expected \(name) rank \(rank), got shape \(shape)."
            ])
        }
        return shape
    }

    private func strides(of array: MLMultiArray) -> [Int] {
        array.strides.map { Int(truncating: $0) }
    }

    private func offset3D(batch: Int, channel: Int, frame: Int, strides: [Int]) -> Int {
        batch * strides[0] + channel * strides[1] + frame * strides[2]
    }

    private func floatValue(_ array: MLMultiArray, at offset: Int) -> Float {
        array[offset].floatValue
    }

    private func resolveLengthArray(
        name: String,
        values: [Int32]?,
        batchSize: Int,
        capacity: Int
    ) throws -> MLMultiArray {
        let resolvedValues = values ?? Array(repeating: Int32(capacity), count: batchSize)
        guard resolvedValues.count == batchSize else {
            throw NSError(domain: "VITSCoreMLDriver", code: 8, userInfo: [
                NSLocalizedDescriptionKey: "\(name) count \(resolvedValues.count) does not match batch size \(batchSize)."
            ])
        }
        for value in resolvedValues {
            if value < 0 || Int(value) > capacity {
                throw NSError(domain: "VITSCoreMLDriver", code: 9, userInfo: [
                    NSLocalizedDescriptionKey: "\(name) value \(value) is outside valid range [0, \(capacity)]."
                ])
            }
        }
        return try makeLengthArray(values: resolvedValues)
    }

    private func sampleLatentHost<R: RandomNumberGenerator>(
        priorMean: MLMultiArray,
        priorLogScale: MLMultiArray,
        noiseScale: Float,
        rng: inout R
    ) throws -> MLMultiArray {
        let meanShape = try requireShape(priorMean, rank: 3, name: "prior_mean")
        let logScaleShape = try requireShape(priorLogScale, rank: 3, name: "prior_log_scale")
        guard meanShape == logScaleShape else {
            throw NSError(domain: "VITSCoreMLDriver", code: 6, userInfo: [
                NSLocalizedDescriptionKey: "prior_mean shape \(meanShape) does not match prior_log_scale shape \(logScaleShape)."
            ])
        }
        let result = try MLMultiArray(shape: priorMean.shape, dataType: .float32)
        let meanStrides = strides(of: priorMean)
        let logStrides = strides(of: priorLogScale)
        let resultStrides = strides(of: result)
        for batchIndex in 0..<meanShape[0] {
            for channelIndex in 0..<meanShape[1] {
                for frameIndex in 0..<meanShape[2] {
                    let meanOffset = offset3D(batch: batchIndex, channel: channelIndex, frame: frameIndex, strides: meanStrides)
                    let logOffset = offset3D(batch: batchIndex, channel: channelIndex, frame: frameIndex, strides: logStrides)
                    let resultOffset = offset3D(batch: batchIndex, channel: channelIndex, frame: frameIndex, strides: resultStrides)
                    let mean = floatValue(priorMean, at: meanOffset)
                    let logScale = floatValue(priorLogScale, at: logOffset)
                    let gaussian = nextGaussian(using: &rng)
                    let sampled = mean + gaussian * Float(exp(Double(logScale))) * noiseScale
                    result[resultOffset] = NSNumber(value: sampled)
                }
            }
        }
        return result
    }

    private func sampleLatentHost(
        priorMean: MLMultiArray,
        priorLogScale: MLMultiArray,
        explicitNoise: MLMultiArray,
        noiseScale: Float
    ) throws -> MLMultiArray {
        let meanShape = try requireShape(priorMean, rank: 3, name: "prior_mean")
        let logScaleShape = try requireShape(priorLogScale, rank: 3, name: "prior_log_scale")
        let noiseShape = try requireShape(explicitNoise, rank: 3, name: "noise")
        guard meanShape == logScaleShape, meanShape == noiseShape else {
            throw NSError(domain: "VITSCoreMLDriver", code: 11, userInfo: [
                NSLocalizedDescriptionKey: "prior_mean/prior_log_scale/noise shape mismatch: \(meanShape), \(logScaleShape), \(noiseShape)."
            ])
        }
        let result = try MLMultiArray(shape: priorMean.shape, dataType: .float32)
        let meanStrides = strides(of: priorMean)
        let logStrides = strides(of: priorLogScale)
        let noiseStrides = strides(of: explicitNoise)
        let resultStrides = strides(of: result)
        for batchIndex in 0..<meanShape[0] {
            for channelIndex in 0..<meanShape[1] {
                for frameIndex in 0..<meanShape[2] {
                    let meanOffset = offset3D(batch: batchIndex, channel: channelIndex, frame: frameIndex, strides: meanStrides)
                    let logOffset = offset3D(batch: batchIndex, channel: channelIndex, frame: frameIndex, strides: logStrides)
                    let noiseOffset = offset3D(batch: batchIndex, channel: channelIndex, frame: frameIndex, strides: noiseStrides)
                    let resultOffset = offset3D(batch: batchIndex, channel: channelIndex, frame: frameIndex, strides: resultStrides)
                    let mean = floatValue(priorMean, at: meanOffset)
                    let logScale = floatValue(priorLogScale, at: logOffset)
                    let gaussian = floatValue(explicitNoise, at: noiseOffset)
                    let sampled = mean + gaussian * Float(exp(Double(logScale))) * noiseScale
                    result[resultOffset] = NSNumber(value: sampled)
                }
            }
        }
        return result
    }

    private func makeGaussianNoiseArray(shape: [Int], seed: UInt64?) throws -> MLMultiArray {
        let noise = try MLMultiArray(shape: shape.map(NSNumber.init(value:)), dataType: .float32)
        if let seed {
            var rng = SplitMix64(seed: seed)
            for index in 0..<noise.count {
                noise[index] = NSNumber(value: nextGaussian(using: &rng))
            }
        } else {
            var rng = SystemRandomNumberGenerator()
            for index in 0..<noise.count {
                noise[index] = NSNumber(value: nextGaussian(using: &rng))
            }
        }
        return noise
    }

    private func resolveNoiseArray(
        explicitNoise: MLMultiArray?,
        expectedShape: [Int],
        seed: UInt64?
    ) throws -> MLMultiArray {
        if let explicitNoise {
            let actualShape = try requireShape(explicitNoise, rank: 3, name: "noise")
            guard actualShape == expectedShape else {
                throw NSError(domain: "VITSCoreMLDriver", code: 12, userInfo: [
                    NSLocalizedDescriptionKey: "Explicit noise shape mismatch. Expected \(expectedShape), got \(actualShape)."
                ])
            }
            return explicitNoise
        }
        return try makeGaussianNoiseArray(shape: expectedShape, seed: seed)
    }

    private func nextGaussian<R: RandomNumberGenerator>(using rng: inout R) -> Float {
        let u1 = max(Double.random(in: 0..<1, using: &rng), 1e-12)
        let u2 = Double.random(in: 0..<1, using: &rng)
        let radius = sqrt(-2.0 * log(u1))
        let theta = 2.0 * Double.pi * u2
        return Float(radius * cos(theta))
    }

    private static func loadModel(at url: URL, configuration: MLModelConfiguration) throws -> MLModel {
        try GPTSoVITSCoreMLModelLoader.loadModel(at: url, configuration: configuration)
    }
}
