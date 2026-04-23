import AVFoundation
import CoreML
import Foundation

public struct ReferenceAudioSamples {
    public let channels: [[Float]]
    public let sampleRate: Int

    public init(channels: [[Float]], sampleRate: Int) throws {
        guard sampleRate > 0 else {
            throw NSError(domain: "ReferenceAudioSamples", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "sampleRate must be positive."
            ])
        }
        guard let first = channels.first else {
            throw NSError(domain: "ReferenceAudioSamples", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "channels must not be empty."
            ])
        }
        guard channels.dropFirst().allSatisfy({ $0.count == first.count }) else {
            throw NSError(domain: "ReferenceAudioSamples", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "All channels must have the same sample count."
            ])
        }
        self.channels = channels
        self.sampleRate = sampleRate
    }

    public init(mono: [Float], sampleRate: Int) throws {
        try self.init(channels: [mono], sampleRate: sampleRate)
    }

    public init(buffer: AVAudioPCMBuffer) throws {
        let converted = try Self.convertToFloat32NonInterleaved(buffer: buffer)
        self = try Self(
            channels: Self.extractChannels(from: converted),
            sampleRate: Int(converted.format.sampleRate.rounded())
        )
    }

    public init(contentsOf url: URL) throws {
        let file = try AVAudioFile(forReading: url)
        guard file.length > 0 else {
            throw NSError(domain: "ReferenceAudioSamples", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "Audio file is empty: \(url.path)"
            ])
        }
        guard let inputBuffer = AVAudioPCMBuffer(
            pcmFormat: file.processingFormat,
            frameCapacity: AVAudioFrameCount(file.length)
        ) else {
            throw NSError(domain: "ReferenceAudioSamples", code: 5, userInfo: [
                NSLocalizedDescriptionKey: "Failed to allocate AVAudioPCMBuffer for file: \(url.path)"
            ])
        }
        try file.read(into: inputBuffer)
        try self.init(buffer: inputBuffer)
    }

    public var channelCount: Int {
        channels.count
    }

    public var sampleCountPerChannel: Int {
        channels.first?.count ?? 0
    }

    private static func convertToFloat32NonInterleaved(
        buffer: AVAudioPCMBuffer
    ) throws -> AVAudioPCMBuffer {
        if
            buffer.format.commonFormat == .pcmFormatFloat32,
            !buffer.format.isInterleaved
        {
            return buffer
        }

        guard let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: buffer.format.sampleRate,
            channels: buffer.format.channelCount,
            interleaved: false
        ) else {
            throw NSError(domain: "ReferenceAudioSamples", code: 6, userInfo: [
                NSLocalizedDescriptionKey: "Failed to create float32 output format."
            ])
        }
        guard let converter = AVAudioConverter(from: buffer.format, to: outputFormat) else {
            throw NSError(domain: "ReferenceAudioSamples", code: 7, userInfo: [
                NSLocalizedDescriptionKey: "Failed to create AVAudioConverter for input buffer."
            ])
        }
        guard let outputBuffer = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: buffer.frameLength
        ) else {
            throw NSError(domain: "ReferenceAudioSamples", code: 8, userInfo: [
                NSLocalizedDescriptionKey: "Failed to allocate float32 output buffer."
            ])
        }

        var didFeedInput = false
        var convertError: NSError?
        let status = converter.convert(to: outputBuffer, error: &convertError) { _, outStatus in
            if didFeedInput {
                outStatus.pointee = .endOfStream
                return nil
            }
            didFeedInput = true
            outStatus.pointee = .haveData
            return buffer
        }

        if let convertError {
            throw convertError
        }
        guard status == .haveData || status == .inputRanDry || status == .endOfStream else {
            throw NSError(domain: "ReferenceAudioSamples", code: 9, userInfo: [
                NSLocalizedDescriptionKey: "Unexpected AVAudioConverter status: \(status.rawValue)"
            ])
        }
        return outputBuffer
    }

    private static func extractChannels(from buffer: AVAudioPCMBuffer) -> [[Float]] {
        let frameLength = Int(buffer.frameLength)
        let channelCount = Int(buffer.format.channelCount)
        guard
            frameLength > 0,
            channelCount > 0,
            let channelData = buffer.floatChannelData
        else {
            return []
        }

        return (0..<channelCount).map { channelIndex in
            Array(
                UnsafeBufferPointer(
                    start: channelData[channelIndex],
                    count: frameLength
                )
            )
        }
    }
}

public struct PreparedReferenceConditioning {
    public let input: ReferenceAudioSamples
    public let features: ExtractedReferenceAudioFeatures
    public let refer: MLMultiArray
    public let speakerFbank80: MLMultiArray
}
