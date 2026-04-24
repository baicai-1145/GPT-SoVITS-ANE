import AVFoundation
import CoreML
import Foundation

public struct CNHubertPreparedInput {
    public let sourceSampleRate: Int
    public let mono: [Float]
    public let resampledMono16k: [Float]
    public let inputValues: [Float]
    public let activeSampleCount: Int
    public let paddedSampleCount: Int
}

public enum CNHubertInputPreparerError: LocalizedError {
    case invalidSampleRate(Int)
    case rawReferenceLengthOutOfRange(actual: Int, lower: Int, upper: Int)
    case activeInputLengthOutOfRange(active: Int, lower: Int, upper: Int)
    case inputLengthExceedsCapacity(active: Int, capacity: Int)

    public var errorDescription: String? {
        switch self {
        case let .invalidSampleRate(sampleRate):
            return "CNHuBERT 输入 sampleRate 必须为正数，收到 \(sampleRate)。"
        case let .rawReferenceLengthOutOfRange(actual, lower, upper):
            return "参考音频重采样到 16k 后长度=\(actual)，不在 Python 版要求的范围 [\(lower), \(upper)] 内。"
        case let .activeInputLengthOutOfRange(active, lower, upper):
            return "CNHuBERT active sample count=\(active)，不在当前 prompt Core ML 导出范围 [\(lower), \(upper)] 内。"
        case let .inputLengthExceedsCapacity(active, capacity):
            return "CNHuBERT active sample count=\(active) 超过当前 prompt Core ML 导出输入容量 \(capacity)。"
        }
    }
}

public final class CNHubertInputPreparer {
    public let targetSampleRate: Int
    public let doNormalize: Bool
    public let trailingSilenceSampleCount: Int
    public let rawReferenceSampleCountRange: ClosedRange<Int>?
    public let activeInputSampleCountRange: ClosedRange<Int>?

    public init(
        targetSampleRate: Int = 16000,
        doNormalize: Bool = true,
        trailingSilenceSampleCount: Int = 0,
        rawReferenceSampleCountRange: ClosedRange<Int>? = nil,
        activeInputSampleCountRange: ClosedRange<Int>? = nil
    ) {
        self.targetSampleRate = targetSampleRate
        self.doNormalize = doNormalize
        self.trailingSilenceSampleCount = max(trailingSilenceSampleCount, 0)
        self.rawReferenceSampleCountRange = rawReferenceSampleCountRange
        self.activeInputSampleCountRange = activeInputSampleCountRange
    }

    private func resolvedSampleRateConverterAlgorithm() -> String {
        switch ProcessInfo.processInfo.environment["GPTSOVITS_RESAMPLE_ALGORITHM"]?.lowercased() {
        case "normal":
            return AVSampleRateConverterAlgorithm_Normal
        case "minimum_phase":
            return AVSampleRateConverterAlgorithm_MinimumPhase
        case "mastering":
            return AVSampleRateConverterAlgorithm_Mastering
        case nil, "":
            return AVSampleRateConverterAlgorithm_Normal
        default:
            return AVSampleRateConverterAlgorithm_Normal
        }
    }

    private func resolvedSampleRateConverterQuality() -> Int {
        guard let raw = ProcessInfo.processInfo.environment["GPTSOVITS_RESAMPLE_QUALITY"],
              let quality = Int(raw) else {
            return 64
        }
        return quality
    }

    private func resolvedPrimeMethod() -> AVAudioConverterPrimeMethod {
        switch ProcessInfo.processInfo.environment["GPTSOVITS_RESAMPLE_PRIME_METHOD"]?.lowercased() {
        case "none":
            return .none
        case "pre":
            return .pre
        case "normal", nil, "":
            return .normal
        default:
            return .normal
        }
    }

    public func prepare(
        referenceAudio: ReferenceAudioSamples,
        paddedSampleCount: Int? = nil
    ) throws -> CNHubertPreparedInput {
        guard referenceAudio.sampleRate > 0 else {
            throw CNHubertInputPreparerError.invalidSampleRate(referenceAudio.sampleRate)
        }
        let mono = try averageToMono(referenceAudio.channels)
        let resampled = try resample(
            mono: mono,
            sourceRate: referenceAudio.sampleRate,
            targetRate: targetSampleRate
        )
        if let rawReferenceSampleCountRange,
           !rawReferenceSampleCountRange.contains(resampled.count) {
            throw CNHubertInputPreparerError.rawReferenceLengthOutOfRange(
                actual: resampled.count,
                lower: rawReferenceSampleCountRange.lowerBound,
                upper: rawReferenceSampleCountRange.upperBound
            )
        }
        let withTrailingSilence: [Float]
        if trailingSilenceSampleCount > 0 {
            withTrailingSilence = resampled + Array(repeating: 0, count: trailingSilenceSampleCount)
        } else {
            withTrailingSilence = resampled
        }
        let normalized = doNormalize ? normalizeZeroMeanUnitVariance(withTrailingSilence) : withTrailingSilence
        let activeSampleCount = normalized.count
        if let activeInputSampleCountRange,
           !activeInputSampleCountRange.contains(activeSampleCount) {
            throw CNHubertInputPreparerError.activeInputLengthOutOfRange(
                active: activeSampleCount,
                lower: activeInputSampleCountRange.lowerBound,
                upper: activeInputSampleCountRange.upperBound
            )
        }
        let inputValues: [Float]
        let resolvedPaddedSampleCount: Int
        if let paddedSampleCount {
            guard activeSampleCount <= paddedSampleCount else {
                throw CNHubertInputPreparerError.inputLengthExceedsCapacity(
                    active: activeSampleCount,
                    capacity: paddedSampleCount
                )
            }
            inputValues = normalized + Array(repeating: 0, count: paddedSampleCount - activeSampleCount)
            resolvedPaddedSampleCount = paddedSampleCount
        } else {
            inputValues = normalized
            resolvedPaddedSampleCount = activeSampleCount
        }

        return CNHubertPreparedInput(
            sourceSampleRate: referenceAudio.sampleRate,
            mono: mono,
            resampledMono16k: resampled,
            inputValues: inputValues,
            activeSampleCount: activeSampleCount,
            paddedSampleCount: resolvedPaddedSampleCount
        )
    }

    public func makeInputValuesMultiArray(
        _ prepared: CNHubertPreparedInput,
        driver: CNHubertCoreMLDriver
    ) throws -> MLMultiArray {
        try driver.makeFloat32Array(shape: [1, prepared.inputValues.count], values: prepared.inputValues)
    }

    private func averageToMono(_ channels: [[Float]]) throws -> [Float] {
        guard let first = channels.first else { return [] }
        guard channels.dropFirst().allSatisfy({ $0.count == first.count }) else {
            throw NSError(domain: "CNHubertInputPreparer", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "All channels must have the same sample count."
            ])
        }
        if channels.count == 1 {
            return first
        }
        let scale = 1.0 / Float(channels.count)
        var mono = Array(repeating: Float.zero, count: first.count)
        for channel in channels {
            for index in 0..<channel.count {
                mono[index] += channel[index] * scale
            }
        }
        return mono
    }

    private func normalizeZeroMeanUnitVariance(_ samples: [Float]) -> [Float] {
        guard !samples.isEmpty else { return samples }
        let mean = samples.reduce(0, +) / Float(samples.count)
        let variance = samples.reduce(0) { partial, value in
            let centered = value - mean
            return partial + centered * centered
        } / Float(samples.count)
        let scale = 1.0 / sqrt(max(variance, 1e-7))
        return samples.map { ($0 - mean) * scale }
    }

    private func resample(mono: [Float], sourceRate: Int, targetRate: Int) throws -> [Float] {
        if sourceRate == targetRate || mono.isEmpty {
            return mono
        }
        if resolvedResamplerBackend() == "soxr" {
            do {
                return try GPTSoVITSSoXRResampler.resample(
                    mono: mono,
                    sourceRate: sourceRate,
                    targetRate: targetRate
                )
            } catch {
                if ProcessInfo.processInfo.environment["GPTSOVITS_SOXR_REQUIRED"] == "1" {
                    throw error
                }
            }
        }

        guard
            let sourceFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: Double(sourceRate),
                channels: 1,
                interleaved: false
            ),
            let targetFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: Double(targetRate),
                channels: 1,
                interleaved: false
            ),
            let inputBuffer = AVAudioPCMBuffer(
                pcmFormat: sourceFormat,
                frameCapacity: AVAudioFrameCount(mono.count)
            )
        else {
            throw NSError(domain: "CNHubertInputPreparer", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Failed to create AVAudioFormat/AVAudioPCMBuffer for resampling."
            ])
        }

        inputBuffer.frameLength = AVAudioFrameCount(mono.count)
        let inputPointer = inputBuffer.floatChannelData![0]
        for (index, sample) in mono.enumerated() {
            inputPointer[index] = sample
        }

        let outputCapacity = AVAudioFrameCount(ceil(Double(mono.count) * Double(targetRate) / Double(sourceRate))) + 16
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outputCapacity) else {
            throw NSError(domain: "CNHubertInputPreparer", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "Failed to allocate output buffer for resampling."
            ])
        }

        guard let converter = AVAudioConverter(from: sourceFormat, to: targetFormat) else {
            throw NSError(domain: "CNHubertInputPreparer", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "Failed to create AVAudioConverter."
            ])
        }
        converter.sampleRateConverterQuality = resolvedSampleRateConverterQuality()
        converter.sampleRateConverterAlgorithm = resolvedSampleRateConverterAlgorithm()
        converter.primeMethod = resolvedPrimeMethod()

        var didFeedInput = false
        var convertError: NSError?
        let status = converter.convert(to: outputBuffer, error: &convertError) { _, outStatus in
            if didFeedInput {
                outStatus.pointee = .endOfStream
                return nil
            }
            didFeedInput = true
            outStatus.pointee = .haveData
            return inputBuffer
        }

        if let convertError {
            throw convertError
        }
        guard status == .haveData || status == .inputRanDry || status == .endOfStream else {
            throw NSError(domain: "CNHubertInputPreparer", code: 5, userInfo: [
                NSLocalizedDescriptionKey: "Unexpected AVAudioConverter status: \(status.rawValue)"
            ])
        }

        let frameLength = Int(outputBuffer.frameLength)
        let outputPointer = outputBuffer.floatChannelData![0]
        return Array(UnsafeBufferPointer(start: outputPointer, count: frameLength))
    }

    private func resolvedResamplerBackend() -> String {
        switch ProcessInfo.processInfo.environment["GPTSOVITS_RESAMPLER"]?.lowercased() {
        case "avaudio":
            return "avaudio"
        case "soxr", nil, "":
            return GPTSoVITSSoXRResampler.isAvailable() ? "soxr" : "avaudio"
        default:
            return GPTSoVITSSoXRResampler.isAvailable() ? "soxr" : "avaudio"
        }
    }
}
