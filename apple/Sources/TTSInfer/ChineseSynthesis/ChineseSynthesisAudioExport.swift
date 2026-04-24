import AVFoundation
import CoreML
import Foundation

public struct GPTSoVITSChineseSynthesisSegmentAudioExport {
    public let samples: [Float]
    public let activeFrameCount: Int
    public let activeSampleCount: Int
    public let totalSampleCount: Int
    public let trailingPaddingSampleCount: Int
    public let sampleStride: Int
}

public struct GPTSoVITSChineseSynthesisAudioExport {
    public let samples: [Float]
    public let outputSampleRate: Double
    public let hopLength: Int
    public let gapSampleCount: Int
    public let segmentExports: [GPTSoVITSChineseSynthesisSegmentAudioExport]
}

public enum GPTSoVITSChineseSynthesisAudioExportError: LocalizedError {
    case invalidOutputPath(String)

    public var errorDescription: String? {
        switch self {
        case let .invalidOutputPath(path):
            return "Invalid output path: \(path)"
        }
    }
}

public enum GPTSoVITSChineseSynthesisAudioExporter {
    public static func export(
        from result: GPTSoVITSChineseSynthesisResult,
        pipeline: GPTSoVITSChineseSynthesisPipeline,
        fragmentIntervalSeconds: Double = 0.3
    ) -> GPTSoVITSChineseSynthesisAudioExport {
        let referSpectrogramContract = pipeline.speakerConditioned.vits.vits.manifest.runtime.referenceAudioContract?.referSpectrogramContract
        let outputSampleRate = Double(referSpectrogramContract?.targetSampleRate ?? 32000)
        let hopLength = referSpectrogramContract?.hopLength ?? 640
        let gapSampleCount = max(Int(outputSampleRate * fragmentIntervalSeconds), 0)
        let concatenated = concatenatedSegmentAudio(
            segments: result.segments,
            hopLength: hopLength,
            gapSampleCount: gapSampleCount
        )
        return GPTSoVITSChineseSynthesisAudioExport(
            samples: concatenated.samples,
            outputSampleRate: outputSampleRate,
            hopLength: hopLength,
            gapSampleCount: gapSampleCount,
            segmentExports: concatenated.segmentExports
        )
    }

    public static func writeWAV(
        samples: [Float],
        sampleRate: Double,
        to outputURL: URL
    ) throws {
        let parent = outputURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: parent,
            withIntermediateDirectories: true,
            attributes: nil
        )

        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw GPTSoVITSChineseSynthesisAudioExportError.invalidOutputPath(outputURL.path)
        }
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(samples.count)
        ) else {
            throw GPTSoVITSChineseSynthesisAudioExportError.invalidOutputPath(outputURL.path)
        }

        buffer.frameLength = AVAudioFrameCount(samples.count)
        let channel = buffer.floatChannelData![0]
        for (index, sample) in samples.enumerated() {
            channel[index] = max(-1.0, min(1.0, sample))
        }

        let audioFile = try AVAudioFile(
            forWriting: outputURL,
            settings: format.settings,
            commonFormat: .pcmFormatFloat32,
            interleaved: false
        )
        try audioFile.write(from: buffer)
    }

    private static func concatenatedSegmentAudio(
        segments: [GPTSoVITSChineseSynthesisSegmentResult],
        hopLength: Int,
        gapSampleCount: Int
    ) -> (samples: [Float], segmentExports: [GPTSoVITSChineseSynthesisSegmentAudioExport]) {
        var output = [Float]()
        var segmentExports = [GPTSoVITSChineseSynthesisSegmentAudioExport]()
        output.reserveCapacity(segments.count * max(gapSampleCount, 0))
        segmentExports.reserveCapacity(segments.count)

        for (index, segment) in segments.enumerated() {
            let segmentExport = trimmedSegmentAudio(segment, fallbackHopLength: hopLength)
            segmentExports.append(segmentExport)
            output.append(contentsOf: segmentExport.samples)
            if index + 1 < segments.count, gapSampleCount > 0 {
                output.append(contentsOf: Array(repeating: 0, count: gapSampleCount))
            }
        }

        return (output, segmentExports)
    }

    private static func trimmedSegmentAudio(
        _ segment: GPTSoVITSChineseSynthesisSegmentResult,
        fallbackHopLength: Int
    ) -> GPTSoVITSChineseSynthesisSegmentAudioExport {
        let fullAudio = floatValues(segment.synthesis.vitsResult.audio)
        let activeFrameCount = leadingActiveFrameCount(in: segment.synthesis.vitsResult.prior.yMask)
        let sampleStride = inferredAudioSampleStride(
            audio: segment.synthesis.vitsResult.audio,
            yMask: segment.synthesis.vitsResult.prior.yMask,
            fallbackHopLength: fallbackHopLength
        )
        let unclampedActiveSampleCount = activeFrameCount * sampleStride
        let activeSampleCount = min(unclampedActiveSampleCount, fullAudio.count)
        let trimmedSamples = Array(fullAudio.prefix(activeSampleCount))
        return GPTSoVITSChineseSynthesisSegmentAudioExport(
            samples: trimmedSamples,
            activeFrameCount: activeFrameCount,
            activeSampleCount: activeSampleCount,
            totalSampleCount: fullAudio.count,
            trailingPaddingSampleCount: max(fullAudio.count - activeSampleCount, 0),
            sampleStride: sampleStride
        )
    }

    private static func leadingActiveFrameCount(in yMask: MLMultiArray) -> Int {
        var count = 0
        for value in floatValues(yMask) {
            if value > 0.5 {
                count += 1
            } else {
                break
            }
        }
        return count
    }

    private static func inferredAudioSampleStride(
        audio: MLMultiArray,
        yMask: MLMultiArray,
        fallbackHopLength: Int
    ) -> Int {
        let audioShape = formatShape(audio)
        let maskShape = formatShape(yMask)
        guard audioShape.count == 3, maskShape.count == 3 else {
            return max(fallbackHopLength, 1)
        }

        let waveformSampleCount = audioShape[2]
        let latentFrameCount = maskShape[2]
        guard waveformSampleCount > 0,
              latentFrameCount > 0,
              waveformSampleCount % latentFrameCount == 0 else {
            return max(fallbackHopLength, 1)
        }
        return waveformSampleCount / latentFrameCount
    }

    private static func floatValues(_ array: MLMultiArray) -> [Float] {
        (0..<array.count).map { index in
            Float(truncating: array[index])
        }
    }

    private static func formatShape(_ array: MLMultiArray) -> [Int] {
        array.shape.map { Int(truncating: $0) }
    }
}
