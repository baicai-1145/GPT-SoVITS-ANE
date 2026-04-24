import CoreML
import Foundation

public struct GPTSoVITSPromptSemanticResult {
    public let preparedInput: CNHubertPreparedInput
    public let inputValues: MLMultiArray
    public let sslContent: MLMultiArray
    public let promptSemantic: MLMultiArray
    public let promptCount: Int
    public let prompts: [Int32]
}

public enum GPTSoVITSPromptSemanticPipelineError: LocalizedError {
    case missingFixedShape(String)
    case invalidShape(String)
    case promptCountExceedsCapacity(promptCount: Int, capacity: Int)

    public var errorDescription: String? {
        switch self {
        case let .missingFixedShape(name):
            return "Prompt semantic 模型输入 \(name) 没有固定 multi-array shape。"
        case let .invalidShape(name):
            return "Prompt semantic 模型输入 \(name) 的 shape 不符合当前预期。"
        case let .promptCountExceedsCapacity(promptCount, capacity):
            return "Prompt token count=\(promptCount) 超过当前 prompt Core ML 导出容量 \(capacity)。"
        }
    }
}

public final class GPTSoVITSPromptSemanticPipeline {
    public let cnhubert: CNHubertCoreMLDriver
    public let sslLatentExtractor: SSLLatentExtractorCoreMLDriver
    public let inputPreparer: CNHubertInputPreparer
    public let bundle: PromptSemanticCoreMLDriver?

    public init(
        cnhubert: CNHubertCoreMLDriver,
        sslLatentExtractor: SSLLatentExtractorCoreMLDriver,
        inputPreparer: CNHubertInputPreparer = CNHubertInputPreparer(),
        bundle: PromptSemanticCoreMLDriver? = nil
    ) {
        self.cnhubert = cnhubert
        self.sslLatentExtractor = sslLatentExtractor
        self.inputPreparer = inputPreparer
        self.bundle = bundle
    }

    public convenience init(
        cnhubertModelURL: URL,
        sslLatentExtractorModelURL: URL,
        configuration: MLModelConfiguration = MLModelConfiguration(),
        inputPreparer: CNHubertInputPreparer = CNHubertInputPreparer()
    ) throws {
        self.init(
            cnhubert: try CNHubertCoreMLDriver(
                modelURL: cnhubertModelURL,
                configuration: configuration
            ),
            sslLatentExtractor: try SSLLatentExtractorCoreMLDriver(
                modelURL: sslLatentExtractorModelURL,
                configuration: configuration
            ),
            inputPreparer: inputPreparer
        )
    }

    public convenience init(
        bundleDirectory: URL,
        configuration: MLModelConfiguration = MLModelConfiguration()
    ) throws {
        let bundle = try PromptSemanticCoreMLDriver(
            bundleDirectory: bundleDirectory,
            configuration: configuration
        )
        self.init(
            cnhubert: bundle.cnhubert,
            sslLatentExtractor: bundle.sslLatentExtractor,
            inputPreparer: CNHubertInputPreparer(
                targetSampleRate: bundle.manifest.runtime.audioInputContract.targetSampleRate,
                doNormalize: bundle.manifest.runtime.audioInputContract.normalization.doNormalize,
                trailingSilenceSampleCount: bundle.manifest.runtime.audioInputContract.trailingSilenceSampleCount,
                rawReferenceSampleCountRange: bundle.manifest.runtime.audioInputContract
                    .rawReferenceSampleCountRange?
                    .closedRange,
                activeInputSampleCountRange: bundle.manifest.runtime.audioInputContract
                    .activeInputSampleCountRange?
                    .closedRange
            ),
            bundle: bundle
        )
    }

    public func extractPrompts(referenceAudio: ReferenceAudioSamples) throws -> GPTSoVITSPromptSemanticResult {
        let preparedInput = try inputPreparer.prepare(
            referenceAudio: referenceAudio,
            paddedSampleCount: resolvedPaddedSampleCount()
        )
        let inputValues = try inputPreparer.makeInputValuesMultiArray(preparedInput, driver: cnhubert)
        let sslContent = try cnhubert.predictSSLContent(inputValues: inputValues)
        try validateSSLShape(sslContent)
        let promptSemantic = try sslLatentExtractor.predictPromptSemantic(sslContent: sslContent)
        let promptCount = resolvePromptCount(preparedInput: preparedInput, promptSemantic: promptSemantic)
        let prompts = (0..<promptCount).map { Int32(truncating: promptSemantic[$0]) }
        if let fixedPromptCount = sslLatentExtractor.fixedPromptCount, promptCount > fixedPromptCount {
            throw GPTSoVITSPromptSemanticPipelineError.promptCountExceedsCapacity(
                promptCount: promptCount,
                capacity: fixedPromptCount
            )
        }
        return GPTSoVITSPromptSemanticResult(
            preparedInput: preparedInput,
            inputValues: inputValues,
            sslContent: sslContent,
            promptSemantic: promptSemantic,
            promptCount: promptCount,
            prompts: prompts
        )
    }

    private func resolvePromptCount(
        preparedInput: CNHubertPreparedInput,
        promptSemantic: MLMultiArray
    ) -> Int {
        let promptHopSamples = bundle?.manifest.runtime.shapes.promptHopSamples ?? 640
        guard promptHopSamples > 0 else {
            return promptSemantic.count
        }
        let inferredPromptCount = preparedInput.activeSampleCount / promptHopSamples
        let fixedPromptCount = bundle?.manifest.runtime.shapes.promptLen ?? sslLatentExtractor.fixedPromptCount ?? promptSemantic.count
        return max(0, min(promptSemantic.count, min(inferredPromptCount, fixedPromptCount)))
    }

    private func resolvedPaddedSampleCount() -> Int? {
        guard !usesDynamicInputSampleRange() else {
            return nil
        }
        return cnhubert.fixedInputSampleCount
    }

    private func usesDynamicInputSampleRange() -> Bool {
        bundle?.manifest.runtime.shapes.inputSampleCountRange != nil
    }

    private func usesDynamicSSLFrameRange() -> Bool {
        bundle?.manifest.runtime.shapes.sslFrameRange != nil
    }

    private func validateSSLShape(_ sslContent: MLMultiArray) throws {
        if usesDynamicSSLFrameRange() {
            return
        }
        guard let expectedShape = sslLatentExtractor.fixedInputShape else {
            throw GPTSoVITSPromptSemanticPipelineError.missingFixedShape("ssl_content")
        }
        let actualShape = sslContent.shape.map { Int(truncating: $0) }
        guard actualShape == expectedShape else {
            throw GPTSoVITSPromptSemanticPipelineError.invalidShape("ssl_content")
        }
    }
}
