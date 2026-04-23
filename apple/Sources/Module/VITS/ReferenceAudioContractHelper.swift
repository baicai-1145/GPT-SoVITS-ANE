import Foundation

public struct PreparedReferenceAudio {
    public let normalizedMono: [Float]
    public let referWaveform: [Float]
    public let speakerWaveform: [Float]
    public let referSpectrogramShape: [Int]
    public let speakerFbankShape: [Int]
}

public final class VITSReferenceAudioContractHelper {
    public let contract: VITSBundleManifest.Runtime.ReferenceAudioContract
    private static let resampleKernelCacheLock = NSLock()
    private static var resampleKernelCache = [String: SincResampleKernel]()

    public init(contract: VITSBundleManifest.Runtime.ReferenceAudioContract) {
        self.contract = contract
    }

    public func prepareReferenceAudio(
        channels: [[Float]],
        sampleRate: Int
    ) throws -> PreparedReferenceAudio {
        try prepareReferenceAudio(
            referenceAudio: ReferenceAudioSamples(
                channels: channels,
                sampleRate: sampleRate
            )
        )
    }

    public func prepareReferenceAudio(
        referenceAudio: ReferenceAudioSamples
    ) throws -> PreparedReferenceAudio {
        let mono = try averageToMono(referenceAudio.channels)
        let normalized = normalizeAmplitudeIfNeeded(mono)
        let referWaveform = try resample(
            mono: normalized,
            sourceRate: referenceAudio.sampleRate,
            targetRate: contract.referSpectrogramContract.targetSampleRate
        )
        let speakerWaveform = try resample(
            mono: normalized,
            sourceRate: referenceAudio.sampleRate,
            targetRate: contract.speakerFbank80Contract.targetSampleRate
        )
        return PreparedReferenceAudio(
            normalizedMono: normalized,
            referWaveform: referWaveform,
            speakerWaveform: speakerWaveform,
            referSpectrogramShape: referSpectrogramShape(forSampleCount: referWaveform.count),
            speakerFbankShape: speakerFbankShape(forSampleCount: speakerWaveform.count)
        )
    }

    public func averageToMono(_ channels: [[Float]]) throws -> [Float] {
        guard let first = channels.first else {
            return []
        }
        guard channels.dropFirst().allSatisfy({ $0.count == first.count }) else {
            throw NSError(domain: "VITSReferenceAudioContractHelper", code: 1, userInfo: [
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

    public func normalizeAmplitudeIfNeeded(_ mono: [Float]) -> [Float] {
        guard let maxAbs = mono.map({ abs($0) }).max(), maxAbs > 1.0 else {
            return mono
        }
        let divisor = min(2.0 as Float, maxAbs)
        return mono.map { $0 / divisor }
    }

    public func referSpectrogramShape(forSampleCount sampleCount: Int) -> [Int] {
        let hopLength = contract.referSpectrogramContract.hopLength
        let frameCount = sampleCount >= hopLength ? sampleCount / hopLength : 0
        return [1, contract.referSpectrogramContract.expectedFrequencyBins, frameCount]
    }

    public func speakerFbankShape(forSampleCount sampleCount: Int) -> [Int] {
        let sampleRate = contract.speakerFbank80Contract.targetSampleRate
        let windowSize = Int(Double(sampleRate) * 0.025)
        let windowShift = Int(Double(sampleRate) * 0.010)
        let frameCount = sampleCount >= windowSize ? 1 + (sampleCount - windowSize) / windowShift : 0
        return [1, frameCount, contract.speakerFbank80Contract.numMelBins]
    }

    public func resample(mono: [Float], sourceRate: Int, targetRate: Int) throws -> [Float] {
        guard sourceRate > 0, targetRate > 0 else {
            throw NSError(domain: "VITSReferenceAudioContractHelper", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Sample rates must be positive."
            ])
        }
        if sourceRate == targetRate || mono.isEmpty {
            return mono
        }
        let kernel = Self.cachedSincResampleKernel(sourceRate: sourceRate, targetRate: targetRate)
        return Self.applySincResampleKernel(mono: mono, kernel: kernel)
    }

    private static func cachedSincResampleKernel(sourceRate: Int, targetRate: Int) -> SincResampleKernel {
        let key = "\(sourceRate)->\(targetRate)"
        resampleKernelCacheLock.lock()
        if let cached = resampleKernelCache[key] {
            resampleKernelCacheLock.unlock()
            return cached
        }
        resampleKernelCacheLock.unlock()

        let built = buildSincResampleKernel(sourceRate: sourceRate, targetRate: targetRate)
        resampleKernelCacheLock.lock()
        resampleKernelCache[key] = built
        resampleKernelCacheLock.unlock()
        return built
    }

    private static func buildSincResampleKernel(sourceRate: Int, targetRate: Int) -> SincResampleKernel {
        let gcd = greatestCommonDivisor(sourceRate, targetRate)
        let reducedSourceRate = sourceRate / gcd
        let reducedTargetRate = targetRate / gcd
        let lowpassFilterWidth = 6
        let rolloff = 0.99
        let baseFrequency = Double(min(reducedSourceRate, reducedTargetRate)) * rolloff
        let width = Int(ceil(Double(lowpassFilterWidth * reducedSourceRate) / baseFrequency))
        let tapCount = (2 * width) + reducedSourceRate
        let scale = baseFrequency / Double(reducedSourceRate)
        let lowpassWidth = Double(lowpassFilterWidth)
        let denominator = Double(reducedSourceRate)
        let targetDenominator = Double(reducedTargetRate)

        let kernels = (0..<reducedTargetRate).map { phaseIndex in
            let phaseOffset = -Double(phaseIndex) / targetDenominator
            return (0..<tapCount).map { tapIndex -> Float in
                let sampleOffset = Double(tapIndex - width) / denominator
                let clampedT = max(
                    -lowpassWidth,
                    min((phaseOffset + sampleOffset) * baseFrequency, lowpassWidth)
                )
                let window = pow(cos(clampedT * .pi / lowpassWidth / 2.0), 2.0)
                let radians = clampedT * .pi
                let sinc = radians == 0 ? 1.0 : sin(radians) / radians
                return Float(sinc * window * scale)
            }
        }

        return SincResampleKernel(
            reducedSourceRate: reducedSourceRate,
            reducedTargetRate: reducedTargetRate,
            width: width,
            kernels: kernels
        )
    }

    private static func applySincResampleKernel(
        mono: [Float],
        kernel: SincResampleKernel
    ) -> [Float] {
        let padded = Array(repeating: Float.zero, count: kernel.width)
            + mono
            + Array(repeating: Float.zero, count: kernel.width + kernel.reducedSourceRate)
        let stepCount = max((padded.count - kernel.tapCount) / kernel.reducedSourceRate + 1, 0)
        let targetLength = Int(
            ceil(Double(kernel.reducedTargetRate * mono.count) / Double(kernel.reducedSourceRate))
        )
        var output = [Float]()
        output.reserveCapacity(stepCount * kernel.reducedTargetRate)

        for stepIndex in 0..<stepCount {
            let baseIndex = stepIndex * kernel.reducedSourceRate
            for phaseIndex in 0..<kernel.reducedTargetRate {
                let taps = kernel.kernels[phaseIndex]
                var value = 0.0
                for tapIndex in 0..<kernel.tapCount {
                    value += Double(padded[baseIndex + tapIndex]) * Double(taps[tapIndex])
                }
                output.append(Float(value))
            }
        }

        if output.count > targetLength {
            output.removeSubrange(targetLength..<output.count)
        }
        return output
    }

    private static func greatestCommonDivisor(_ lhs: Int, _ rhs: Int) -> Int {
        var a = abs(lhs)
        var b = abs(rhs)
        while b != 0 {
            let remainder = a % b
            a = b
            b = remainder
        }
        return max(a, 1)
    }
}

private struct SincResampleKernel {
    let reducedSourceRate: Int
    let reducedTargetRate: Int
    let width: Int
    let kernels: [[Float]]

    var tapCount: Int {
        kernels.first?.count ?? 0
    }
}
