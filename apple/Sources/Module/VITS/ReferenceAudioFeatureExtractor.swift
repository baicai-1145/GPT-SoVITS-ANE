import Accelerate
import CoreML
import Foundation

public struct ExtractedReferenceAudioFeatures {
    public let preparedAudio: PreparedReferenceAudio
    public let referSpectrogram: [Float]
    public let referSpectrogramShape: [Int]
    public let speakerFbank80: [Float]
    public let speakerFbank80Shape: [Int]

    public func makeReferSpectrogramMultiArray() throws -> MLMultiArray {
        try Self.makeFloat32Array(shape: referSpectrogramShape, values: referSpectrogram)
    }

    public func makeSpeakerFbank80MultiArray() throws -> MLMultiArray {
        try Self.makeFloat32Array(shape: speakerFbank80Shape, values: speakerFbank80)
    }

    private static func makeFloat32Array(shape: [Int], values: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map(NSNumber.init(value:)), dataType: .float32)
        guard array.count == values.count else {
            throw NSError(domain: "ExtractedReferenceAudioFeatures", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Value count \(values.count) does not match shape element count \(array.count)."
            ])
        }
        for (index, value) in values.enumerated() {
            array[index] = NSNumber(value: value)
        }
        return array
    }
}

public final class VITSReferenceAudioFeatureExtractor {
    private let helper: VITSReferenceAudioContractHelper
    private let referContract: VITSBundleManifest.Runtime.ReferenceAudioContract.ReferSpectrogramContract
    private let speakerContract: VITSBundleManifest.Runtime.ReferenceAudioContract.SpeakerFbankContract
    private let referWindow: [Float]
    private let speakerWindow: [Float]
    private let referFFTPlan: RealDFTPlan
    private let speakerFFTPlan: RealDFTPlan
    private let speakerMelFilterBank: [[Float]]
    private let speakerEpsilon: Float
    private let speakerWindowSize: Int
    private let speakerWindowShift: Int
    private let speakerPaddedWindowSize: Int

    public init(contract: VITSBundleManifest.Runtime.ReferenceAudioContract) throws {
        self.helper = VITSReferenceAudioContractHelper(contract: contract)
        self.referContract = contract.referSpectrogramContract
        self.speakerContract = contract.speakerFbank80Contract
        self.referWindow = Self.makePeriodicHannWindow(count: contract.referSpectrogramContract.winLength)
        self.speakerWindowSize = Int(Double(contract.speakerFbank80Contract.targetSampleRate) * 0.025)
        self.speakerWindowShift = Int(Double(contract.speakerFbank80Contract.targetSampleRate) * 0.010)
        self.speakerPaddedWindowSize = Self.nextPowerOfTwo(max(speakerWindowSize, 1))
        self.speakerWindow = Self.makePoveyWindow(count: speakerWindowSize)
        self.referFFTPlan = RealDFTPlan(fftSize: contract.referSpectrogramContract.nFFT)
        self.speakerFFTPlan = RealDFTPlan(fftSize: speakerPaddedWindowSize)
        self.speakerMelFilterBank = try Self.makeKaldiMelFilterBank(
            numMelBins: contract.speakerFbank80Contract.numMelBins,
            paddedWindowSize: speakerPaddedWindowSize,
            sampleFrequency: Float(contract.speakerFbank80Contract.targetSampleRate),
            lowFrequency: 20.0,
            highFrequency: 0.0
        )
        self.speakerEpsilon = Float.ulpOfOne
    }

    public func extractFeatures(
        channels: [[Float]],
        sampleRate: Int
    ) throws -> ExtractedReferenceAudioFeatures {
        try extractFeatures(
            referenceAudio: ReferenceAudioSamples(
                channels: channels,
                sampleRate: sampleRate
            )
        )
    }

    public func extractFeatures(
        referenceAudio: ReferenceAudioSamples
    ) throws -> ExtractedReferenceAudioFeatures {
        let prepared = try helper.prepareReferenceAudio(referenceAudio: referenceAudio)
        let referShape = helper.referSpectrogramShape(forSampleCount: prepared.referWaveform.count)
        let speakerShape = helper.speakerFbankShape(forSampleCount: prepared.speakerWaveform.count)
        let referSpectrogram = try makeReferSpectrogram(
            waveform: prepared.referWaveform,
            expectedShape: referShape
        )
        let speakerFbank80 = try makeSpeakerFbank80(
            waveform: prepared.speakerWaveform,
            expectedShape: speakerShape
        )
        return ExtractedReferenceAudioFeatures(
            preparedAudio: prepared,
            referSpectrogram: referSpectrogram,
            referSpectrogramShape: referShape,
            speakerFbank80: speakerFbank80,
            speakerFbank80Shape: speakerShape
        )
    }

    public func makeReferSpectrogram(
        waveform: [Float],
        expectedShape: [Int]? = nil
    ) throws -> [Float] {
        let pad = max((referContract.nFFT - referContract.hopLength) / 2, 0)
        let padded = Self.reflectPad1D(waveform, left: pad, right: pad)
        let frameCount = padded.count >= referContract.nFFT
            ? 1 + (padded.count - referContract.nFFT) / referContract.hopLength
            : 0
        let binCount = referContract.expectedFrequencyBins
        var output = [Float](repeating: 0, count: binCount * frameCount)

        for frameIndex in 0..<frameCount {
            let frameStart = frameIndex * referContract.hopLength
            var windowed = [Float](repeating: 0, count: referContract.nFFT)
            for offset in 0..<min(referContract.winLength, referContract.nFFT) {
                windowed[offset] = padded[frameStart + offset] * referWindow[offset]
            }
            let spectrum = referFFTPlan.magnitudeSpectrum(
                frame: windowed,
                epsilon: Float(referContract.magnitudeEpsilon)
            )
            for binIndex in 0..<binCount {
                output[binIndex * frameCount + frameIndex] = spectrum[binIndex]
            }
        }

        if let expectedShape {
            let expectedCount = expectedShape.reduce(1, *)
            guard expectedShape == [1, binCount, frameCount], expectedCount == output.count else {
                throw NSError(domain: "VITSReferenceAudioFeatureExtractor", code: 1, userInfo: [
                    NSLocalizedDescriptionKey: "Refer spectrogram shape mismatch. Expected \(expectedShape), got [1, \(binCount), \(frameCount)]."
                ])
            }
        }
        return output
    }

    public func makeSpeakerFbank80(
        waveform: [Float],
        expectedShape: [Int]? = nil
    ) throws -> [Float] {
        let frames = Self.makeSnipEdgeFrames(
            waveform: waveform,
            windowSize: speakerWindowSize,
            windowShift: speakerWindowShift
        )
        let melBinCount = speakerContract.numMelBins
        var output = [Float](repeating: 0, count: frames.count * melBinCount)

        for (frameIndex, frame) in frames.enumerated() {
            let dcRemoved = Self.subtractMean(frame)
            let preemphasized = Self.applyPreemphasis(dcRemoved, coefficient: 0.97)
            var paddedWindow = [Float](repeating: 0, count: speakerPaddedWindowSize)
            for sampleIndex in 0..<speakerWindowSize {
                paddedWindow[sampleIndex] = preemphasized[sampleIndex] * speakerWindow[sampleIndex]
            }
            let powerSpectrum = speakerFFTPlan.magnitudeSquaredSpectrum(frame: paddedWindow)
            for melIndex in 0..<melBinCount {
                let weights = speakerMelFilterBank[melIndex]
                var melEnergy: Float = 0
                for binIndex in 0..<weights.count {
                    melEnergy += powerSpectrum[binIndex] * weights[binIndex]
                }
                output[frameIndex * melBinCount + melIndex] = logf(max(melEnergy, speakerEpsilon))
            }
        }

        if let expectedShape {
            let expectedFrameCount = frames.count
            let expectedCount = expectedShape.reduce(1, *)
            guard expectedShape == [1, expectedFrameCount, melBinCount], expectedCount == output.count else {
                throw NSError(domain: "VITSReferenceAudioFeatureExtractor", code: 2, userInfo: [
                    NSLocalizedDescriptionKey: "Speaker fbank shape mismatch. Expected \(expectedShape), got [1, \(expectedFrameCount), \(melBinCount)]."
                ])
            }
        }
        return output
    }

    private static func nextPowerOfTwo(_ value: Int) -> Int {
        guard value > 1 else { return 1 }
        var power = 1
        while power < value {
            power <<= 1
        }
        return power
    }

    private static func makePeriodicHannWindow(count: Int) -> [Float] {
        guard count > 0 else { return [] }
        if count == 1 { return [1] }
        let scale = 2 * Float.pi / Float(count)
        return (0..<count).map { index in
            0.5 - 0.5 * cosf(scale * Float(index))
        }
    }

    private static func makeSymmetricHannWindow(count: Int) -> [Float] {
        guard count > 0 else { return [] }
        if count == 1 { return [1] }
        let scale = 2 * Float.pi / Float(count - 1)
        return (0..<count).map { index in
            0.5 - 0.5 * cosf(scale * Float(index))
        }
    }

    private static func makePoveyWindow(count: Int) -> [Float] {
        makeSymmetricHannWindow(count: count).map { powf($0, 0.85) }
    }

    private static func reflectPad1D(_ samples: [Float], left: Int, right: Int) -> [Float] {
        guard !samples.isEmpty else { return [] }
        guard samples.count > 1 else { return Array(repeating: samples[0], count: left + samples.count + right) }
        var padded = [Float]()
        padded.reserveCapacity(left + samples.count + right)
        for index in stride(from: left, to: 0, by: -1) {
            padded.append(samples[reflectIndex(-index, count: samples.count)])
        }
        padded.append(contentsOf: samples)
        for index in 0..<right {
            padded.append(samples[reflectIndex(samples.count + index, count: samples.count)])
        }
        return padded
    }

    private static func reflectIndex(_ index: Int, count: Int) -> Int {
        precondition(count > 1, "reflectIndex requires at least 2 samples.")
        var reflected = index
        while reflected < 0 || reflected >= count {
            if reflected < 0 {
                reflected = -reflected
            } else {
                reflected = 2 * count - reflected - 2
            }
        }
        return reflected
    }

    private static func makeSnipEdgeFrames(
        waveform: [Float],
        windowSize: Int,
        windowShift: Int
    ) -> [[Float]] {
        guard windowSize > 0, windowShift > 0, waveform.count >= windowSize else {
            return []
        }
        let frameCount = 1 + (waveform.count - windowSize) / windowShift
        return (0..<frameCount).map { frameIndex in
            let start = frameIndex * windowShift
            return Array(waveform[start..<(start + windowSize)])
        }
    }

    private static func subtractMean(_ values: [Float]) -> [Float] {
        guard !values.isEmpty else { return values }
        let mean = values.reduce(0, +) / Float(values.count)
        return values.map { $0 - mean }
    }

    private static func applyPreemphasis(_ values: [Float], coefficient: Float) -> [Float] {
        guard !values.isEmpty, coefficient != 0 else { return values }
        var output = [Float](repeating: 0, count: values.count)
        output[0] = values[0] - coefficient * values[0]
        if values.count > 1 {
            for index in 1..<values.count {
                output[index] = values[index] - coefficient * values[index - 1]
            }
        }
        return output
    }

    private static func melScale(_ frequency: Float) -> Float {
        1127.0 * logf(1.0 + frequency / 700.0)
    }

    private static func inverseMelScale(_ melFrequency: Float) -> Float {
        700.0 * (expf(melFrequency / 1127.0) - 1.0)
    }

    private static func makeKaldiMelFilterBank(
        numMelBins: Int,
        paddedWindowSize: Int,
        sampleFrequency: Float,
        lowFrequency: Float,
        highFrequency: Float
    ) throws -> [[Float]] {
        guard numMelBins > 3 else {
            throw NSError(domain: "VITSReferenceAudioFeatureExtractor", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "numMelBins must be greater than 3."
            ])
        }
        guard paddedWindowSize % 2 == 0 else {
            throw NSError(domain: "VITSReferenceAudioFeatureExtractor", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "paddedWindowSize must be even."
            ])
        }

        let numFFTBins = paddedWindowSize / 2
        let nyquist = 0.5 * sampleFrequency
        let resolvedHighFrequency = highFrequency <= 0 ? highFrequency + nyquist : highFrequency
        guard lowFrequency >= 0, resolvedHighFrequency > 0, resolvedHighFrequency <= nyquist else {
            throw NSError(domain: "VITSReferenceAudioFeatureExtractor", code: 5, userInfo: [
                NSLocalizedDescriptionKey: "Invalid mel filter bank frequency range."
            ])
        }

        let melLow = melScale(lowFrequency)
        let melHigh = melScale(resolvedHighFrequency)
        let melDelta = (melHigh - melLow) / Float(numMelBins + 1)
        let fftBinWidth = sampleFrequency / Float(paddedWindowSize)
        let melBins = (0..<numFFTBins).map { index in
            melScale(fftBinWidth * Float(index))
        }

        return (0..<numMelBins).map { melIndex in
            let left = melLow + Float(melIndex) * melDelta
            let center = melLow + Float(melIndex + 1) * melDelta
            let right = melLow + Float(melIndex + 2) * melDelta
            var weights = [Float](repeating: 0, count: numFFTBins + 1)
            for fftIndex in 0..<numFFTBins {
                let mel = melBins[fftIndex]
                let weight: Float
                if mel <= left || mel >= right {
                    weight = 0
                } else if mel <= center {
                    weight = (mel - left) / (center - left)
                } else {
                    weight = (right - mel) / (right - center)
                }
                weights[fftIndex] = max(weight, 0)
            }
            weights[numFFTBins] = 0
            _ = inverseMelScale(center)
            return weights
        }
    }
}

private struct RealDFTPlan {
    let fftSize: Int
    let binCount: Int
    let dft: vDSP.DiscreteFourierTransform<Float>

    init(fftSize: Int) {
        precondition(fftSize > 0, "fftSize must be positive.")
        self.fftSize = fftSize
        self.binCount = fftSize / 2 + 1
        do {
            self.dft = try vDSP.DiscreteFourierTransform(
                count: fftSize,
                direction: .forward,
                transformType: .complexComplex,
                ofType: Float.self
            )
        } catch {
            preconditionFailure("Failed to create Accelerate DFT plan for size \(fftSize): \(error)")
        }
    }

    func magnitudeSpectrum(frame: [Float], epsilon: Float) -> [Float] {
        magnitudeLikeSpectrum(frame: frame, epsilon: epsilon, squared: false)
    }

    func magnitudeSquaredSpectrum(frame: [Float]) -> [Float] {
        magnitudeLikeSpectrum(frame: frame, epsilon: 0, squared: true)
    }

    private func magnitudeLikeSpectrum(frame: [Float], epsilon: Float, squared: Bool) -> [Float] {
        precondition(frame.count == fftSize, "frame length must match fftSize")
        let imagInput = [Float](repeating: 0, count: fftSize)
        var realOutput = [Float](repeating: 0, count: fftSize)
        var imagOutput = [Float](repeating: 0, count: fftSize)
        dft.transform(
            inputReal: frame,
            inputImaginary: imagInput,
            outputReal: &realOutput,
            outputImaginary: &imagOutput
        )

        var output = [Float](repeating: 0, count: binCount)
        for binIndex in 0..<binCount {
            let real = realOutput[binIndex]
            let imaginary = imagOutput[binIndex]
            let magnitudeSquared = real * real + imaginary * imaginary
            output[binIndex] = squared ? magnitudeSquared : sqrtf(magnitudeSquared + epsilon)
        }
        return output
    }
}
