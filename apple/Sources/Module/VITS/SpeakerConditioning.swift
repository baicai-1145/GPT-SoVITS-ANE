import CoreML
import Foundation

public struct VITSSpeakerConditioningDriver {
    public let speakerEncoder: SpeakerEncoderCoreMLDriver
    public let vits: VITSCoreMLDriver
    public let referenceAudioFeatureExtractor: VITSReferenceAudioFeatureExtractor?

    public init(
        vitsBundleDirectory: URL,
        configuration: MLModelConfiguration = MLModelConfiguration()
    ) throws {
        let vits = try GPTSoVITSRuntimeProfiler.measure("vits_conditioning.vits_driver.init") {
            try VITSCoreMLDriver(
                bundleDirectory: vitsBundleDirectory,
                configuration: configuration
            )
        }
        guard let speakerEncoderArtifact = vits.manifest.artifacts.speakerEncoder else {
            throw NSError(domain: "VITSSpeakerConditioningDriver", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "The VITS bundle does not include a bundled speaker_encoder artifact."
            ])
        }
        self.vits = vits
        self.speakerEncoder = try GPTSoVITSRuntimeProfiler.measure("vits_conditioning.speaker_encoder.init") {
            try SpeakerEncoderCoreMLDriver(
                modelURL: vitsBundleDirectory.appendingPathComponent(speakerEncoderArtifact.filename),
                configuration: configuration
            )
        }
        if let contract = vits.manifest.runtime.referenceAudioContract {
            self.referenceAudioFeatureExtractor = try GPTSoVITSRuntimeProfiler.measure("vits_conditioning.reference_audio_extractor.init") {
                try VITSReferenceAudioFeatureExtractor(contract: contract)
            }
        } else {
            self.referenceAudioFeatureExtractor = nil
        }
    }

    public init(
        speakerEncoderModelURL: URL,
        vitsBundleDirectory: URL,
        configuration: MLModelConfiguration = MLModelConfiguration()
    ) throws {
        self.speakerEncoder = try GPTSoVITSRuntimeProfiler.measure("vits_conditioning.speaker_encoder.init") {
            try SpeakerEncoderCoreMLDriver(
                modelURL: speakerEncoderModelURL,
                configuration: configuration
            )
        }
        let vits = try GPTSoVITSRuntimeProfiler.measure("vits_conditioning.vits_driver.init") {
            try VITSCoreMLDriver(
                bundleDirectory: vitsBundleDirectory,
                configuration: configuration
            )
        }
        self.vits = vits
        if let contract = vits.manifest.runtime.referenceAudioContract {
            self.referenceAudioFeatureExtractor = try GPTSoVITSRuntimeProfiler.measure("vits_conditioning.reference_audio_extractor.init") {
                try VITSReferenceAudioFeatureExtractor(contract: contract)
            }
        } else {
            self.referenceAudioFeatureExtractor = nil
        }
    }

    public func decodeCondition(
        refer: MLMultiArray,
        speakerFbank80: MLMultiArray
    ) throws -> VITSConditionState {
        let svEmb = try speakerEncoder.embed(fbank80: speakerFbank80)
        return try vits.decodeCondition(refer: refer, svEmb: svEmb)
    }

    public func decode(
        codes: MLMultiArray,
        text: MLMultiArray,
        refer: MLMultiArray,
        speakerFbank80: MLMultiArray,
        noise: MLMultiArray? = nil,
        noiseScale: Float? = nil,
        seed: UInt64? = nil,
        codeLengths: [Int32]? = nil,
        textLengths: [Int32]? = nil
    ) throws -> VITSDecodeResult {
        let svEmb = try speakerEncoder.embed(fbank80: speakerFbank80)
        return try vits.decode(
            codes: codes,
            text: text,
            refer: refer,
            svEmb: svEmb,
            noise: noise,
            noiseScale: noiseScale,
            seed: seed,
            codeLengths: codeLengths,
            textLengths: textLengths
        )
    }

    public func prepareReferenceFeatures(
        channels: [[Float]],
        sampleRate: Int
    ) throws -> ExtractedReferenceAudioFeatures {
        try prepareReferenceFeatures(
            referenceAudio: ReferenceAudioSamples(
                channels: channels,
                sampleRate: sampleRate
            )
        )
    }

    public func prepareReferenceFeatures(
        referenceAudio: ReferenceAudioSamples
    ) throws -> ExtractedReferenceAudioFeatures {
        try requireReferenceAudioFeatureExtractor().extractFeatures(
            referenceAudio: referenceAudio
        )
    }

    public func prepareReferenceConditioning(
        referenceAudio: ReferenceAudioSamples
    ) throws -> PreparedReferenceConditioning {
        let features = try prepareReferenceFeatures(referenceAudio: referenceAudio)
        return PreparedReferenceConditioning(
            input: referenceAudio,
            features: features,
            refer: try makeReferInputMultiArray(from: features),
            speakerFbank80: try makeSpeakerFbank80InputMultiArray(from: features)
        )
    }

    public func decodeCondition(
        referenceAudioChannels channels: [[Float]],
        sampleRate: Int
    ) throws -> VITSConditionState {
        try decodeCondition(
            referenceAudio: ReferenceAudioSamples(
                channels: channels,
                sampleRate: sampleRate
            )
        )
    }

    public func decodeCondition(
        referenceAudio: ReferenceAudioSamples
    ) throws -> VITSConditionState {
        let conditioning = try prepareReferenceConditioning(referenceAudio: referenceAudio)
        return try decodeCondition(
            refer: conditioning.refer,
            speakerFbank80: conditioning.speakerFbank80
        )
    }

    public func decode(
        codes: MLMultiArray,
        text: MLMultiArray,
        referenceAudioChannels channels: [[Float]],
        sampleRate: Int,
        noise: MLMultiArray? = nil,
        noiseScale: Float? = nil,
        seed: UInt64? = nil,
        codeLengths: [Int32]? = nil,
        textLengths: [Int32]? = nil
    ) throws -> VITSDecodeResult {
        try decode(
            codes: codes,
            text: text,
            referenceAudio: ReferenceAudioSamples(
                channels: channels,
                sampleRate: sampleRate
            ),
            noise: noise,
            noiseScale: noiseScale,
            seed: seed,
            codeLengths: codeLengths,
            textLengths: textLengths
        )
    }

    public func decode(
        codes: MLMultiArray,
        text: MLMultiArray,
        referenceAudio: ReferenceAudioSamples,
        noise: MLMultiArray? = nil,
        noiseScale: Float? = nil,
        seed: UInt64? = nil,
        codeLengths: [Int32]? = nil,
        textLengths: [Int32]? = nil
    ) throws -> VITSDecodeResult {
        let conditioning = try prepareReferenceConditioning(referenceAudio: referenceAudio)
        return try decode(
            codes: codes,
            text: text,
            refer: conditioning.refer,
            speakerFbank80: conditioning.speakerFbank80,
            noise: noise,
            noiseScale: noiseScale,
            seed: seed,
            codeLengths: codeLengths,
            textLengths: textLengths
        )
    }

    private func requireReferenceAudioFeatureExtractor() throws -> VITSReferenceAudioFeatureExtractor {
        guard let referenceAudioFeatureExtractor else {
            throw NSError(domain: "VITSSpeakerConditioningDriver", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "The VITS bundle manifest does not contain reference_audio_contract."
            ])
        }
        return referenceAudioFeatureExtractor
    }

    public func makeReferInputMultiArray(
        from features: ExtractedReferenceAudioFeatures
    ) throws -> MLMultiArray {
        let targetShape = try requireDecodeConditionReferShape()
        let targetChannels: Int
        let targetFrames: Int
        if let referModelInputContract = vits.manifest.runtime.referenceAudioContract?.referModelInputContract {
            targetChannels = max(
                min(referModelInputContract.channelSliceEnd, targetShape[1]) - referModelInputContract.channelSliceStart,
                0
            )
            if let frameCountRange = referModelInputContract.frameCountRange {
                let sourceFrames = features.referSpectrogramShape[2]
                guard sourceFrames >= frameCountRange.lowerBound else {
                    throw NSError(domain: "VITSSpeakerConditioningDriver", code: 11, userInfo: [
                        NSLocalizedDescriptionKey:
                            "Reference spectrogram frame count \(sourceFrames) is below decode_condition lower bound \(frameCountRange.lowerBound)."
                    ])
                }
                targetFrames = min(sourceFrames, frameCountRange.upperBound)
            } else {
                targetFrames = min(referModelInputContract.targetFrameLength, targetShape[2])
            }
        } else if let referFrameCountRange = vits.manifest.runtime.shapes.referFrameCountRange {
            let sourceFrames = features.referSpectrogramShape[2]
            guard sourceFrames >= referFrameCountRange.lowerBound else {
                throw NSError(domain: "VITSSpeakerConditioningDriver", code: 12, userInfo: [
                    NSLocalizedDescriptionKey:
                        "Reference spectrogram frame count \(sourceFrames) is below decode_condition lower bound \(referFrameCountRange.lowerBound)."
                ])
            }
            targetChannels = targetShape[1]
            targetFrames = min(sourceFrames, referFrameCountRange.upperBound)
        } else {
            targetChannels = targetShape[1]
            targetFrames = targetShape[2]
        }

        let sourceShape = features.referSpectrogramShape
        let sourceChannels = sourceShape[1]
        let sourceFrames = sourceShape[2]
        let copyChannels = min(targetChannels, sourceChannels)
        let copyFrames = min(targetFrames, sourceFrames)
        let resolvedShape = [1, targetChannels, targetFrames]
        var adapted = [Float](repeating: 0, count: resolvedShape.reduce(1, *))

        for channelIndex in 0..<copyChannels {
            let sourceChannelOffset = channelIndex * sourceFrames
            let targetChannelOffset = channelIndex * targetFrames
            for frameIndex in 0..<copyFrames {
                adapted[targetChannelOffset + frameIndex] = features.referSpectrogram[sourceChannelOffset + frameIndex]
            }
        }

        return try vits.makeFloat32Array(shape: resolvedShape, values: adapted)
    }

    public func makeSpeakerFbank80InputMultiArray(
        from features: ExtractedReferenceAudioFeatures
    ) throws -> MLMultiArray {
        let targetShape = try requireSpeakerFbank80Shape()
        let sourceShape = features.speakerFbank80Shape
        guard sourceShape.count == 3, sourceShape[0] == 1, targetShape.count == 3, targetShape[0] == 1 else {
            throw NSError(domain: "VITSSpeakerConditioningDriver", code: 6, userInfo: [
                NSLocalizedDescriptionKey: "Expected speaker_fbank_80 shape rank 3 with batch size 1."
            ])
        }
        guard sourceShape[2] == targetShape[2] else {
            throw NSError(domain: "VITSSpeakerConditioningDriver", code: 7, userInfo: [
                NSLocalizedDescriptionKey: "Speaker fbank mel bin mismatch. Source \(sourceShape[2]), target \(targetShape[2])."
            ])
        }

        let targetFrames: Int
        if let frameCountRange = vits.manifest.runtime.referenceAudioContract?.speakerFbank80Contract.frameCountRange {
            let sourceFrames = sourceShape[1]
            guard sourceFrames >= frameCountRange.lowerBound else {
                throw NSError(domain: "VITSSpeakerConditioningDriver", code: 13, userInfo: [
                    NSLocalizedDescriptionKey:
                        "Speaker fbank frame count \(sourceFrames) is below speaker_encoder lower bound \(frameCountRange.lowerBound)."
                ])
            }
            targetFrames = min(sourceFrames, frameCountRange.upperBound)
        } else {
            targetFrames = targetShape[1]
        }
        let sourceFrames = sourceShape[1]
        let melBins = targetShape[2]
        let copyFrames = min(sourceFrames, targetFrames)
        let resolvedShape = [1, targetFrames, melBins]
        var adapted = [Float](repeating: 0, count: resolvedShape.reduce(1, *))
        for frameIndex in 0..<copyFrames {
            let sourceOffset = frameIndex * melBins
            let targetOffset = frameIndex * melBins
            for melIndex in 0..<melBins {
                adapted[targetOffset + melIndex] = features.speakerFbank80[sourceOffset + melIndex]
            }
        }
        return try vits.makeFloat32Array(shape: resolvedShape, values: adapted)
    }

    private func requireDecodeConditionReferShape() throws -> [Int] {
        guard let description = vits.decodeConditionModel.modelDescription.inputDescriptionsByName["refer"] else {
            throw NSError(domain: "VITSSpeakerConditioningDriver", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "Missing model input description for 'refer'."
            ])
        }
        guard let constraint = description.multiArrayConstraint else {
            throw NSError(domain: "VITSSpeakerConditioningDriver", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "Input 'refer' does not expose a fixed multi-array shape."
            ])
        }
        let shape = constraint.shape.map { Int(truncating: $0) }
        guard shape.count == 3 else {
            throw NSError(domain: "VITSSpeakerConditioningDriver", code: 5, userInfo: [
                NSLocalizedDescriptionKey: "Expected 'refer' shape rank 3, got \(shape)."
            ])
        }
        return shape
    }

    private func requireSpeakerFbank80Shape() throws -> [Int] {
        guard let description = speakerEncoder.model.modelDescription.inputDescriptionsByName["fbank_80"] else {
            throw NSError(domain: "VITSSpeakerConditioningDriver", code: 8, userInfo: [
                NSLocalizedDescriptionKey: "Missing model input description for 'fbank_80'."
            ])
        }
        guard let constraint = description.multiArrayConstraint else {
            throw NSError(domain: "VITSSpeakerConditioningDriver", code: 9, userInfo: [
                NSLocalizedDescriptionKey: "Input 'fbank_80' does not expose a fixed multi-array shape."
            ])
        }
        let shape = constraint.shape.map { Int(truncating: $0) }
        guard shape.count == 3 else {
            throw NSError(domain: "VITSSpeakerConditioningDriver", code: 10, userInfo: [
                NSLocalizedDescriptionKey: "Expected 'fbank_80' shape rank 3, got \(shape)."
            ])
        }
        return shape
    }
}
