import CoreML
import Foundation

public struct GPTSoVITSChineseSynthesisBundleManifest: Decodable {
    public struct Artifact: Decodable {
        public let target: String
        public let filename: String
        public let path: String
    }

    public struct Artifacts: Decodable {
        public let t2sBundle: Artifact
        public let vitsBundle: Artifact
        public let zhBertChar: Artifact
        public let tokenizer: Artifact
        public let g2pwBundle: Artifact?
        public let promptBundle: Artifact?
        public let englishFrontendBundle: Artifact?
        public let yueFrontendBundle: Artifact?
        public let japaneseFrontendBundle: Artifact?
        public let koreanFrontendBundle: Artifact?

        private enum CodingKeys: String, CodingKey {
            case t2sBundle = "t2s_bundle"
            case vitsBundle = "vits_bundle"
            case zhBertChar = "zh_bert_char"
            case tokenizer
            case g2pwBundle = "g2pw_bundle"
            case promptBundle = "prompt_bundle"
            case englishFrontendBundle = "english_frontend_bundle"
            case yueFrontendBundle = "yue_frontend_bundle"
            case japaneseFrontendBundle = "japanese_frontend_bundle"
            case koreanFrontendBundle = "korean_frontend_bundle"
        }
    }

    public struct Runtime: Decodable {
        public struct Defaults: Decodable {
            public let language: String
            public let splitMethod: String

            private enum CodingKeys: String, CodingKey {
                case language
                case splitMethod = "split_method"
            }
        }

        public struct Features: Decodable {
            public let g2pwFrontend: Bool
            public let promptSemantic: Bool
            public let englishFrontend: Bool?
            public let yueFrontend: Bool?
            public let japaneseFrontend: Bool?
            public let koreanFrontend: Bool?

            private enum CodingKeys: String, CodingKey {
                case g2pwFrontend = "g2pw_frontend"
                case promptSemantic = "prompt_semantic"
                case englishFrontend = "english_frontend"
                case yueFrontend = "yue_frontend"
                case japaneseFrontend = "japanese_frontend"
                case koreanFrontend = "korean_frontend"
            }
        }

        public struct Materialization: Decodable {
            public let mode: String
            public let selfContained: Bool

            private enum CodingKeys: String, CodingKey {
                case mode
                case selfContained = "self_contained"
            }
        }

        public let defaults: Defaults
        public let features: Features
        public let materialization: Materialization?
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

public final class GPTSoVITSChineseSynthesisBundleDriver {
    public let bundleDirectory: URL
    public let manifest: GPTSoVITSChineseSynthesisBundleManifest
    public let pipeline: GPTSoVITSChineseSynthesisPipeline
    public let promptBundleDirectory: URL?
    public let englishFrontendBundleDirectory: URL?
    public let yueFrontendBundleDirectory: URL?
    public let japaneseFrontendBundleDirectory: URL?
    public let koreanFrontendBundleDirectory: URL?
    private let modelConfiguration: MLModelConfiguration

    public init(bundleDirectory: URL, configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        self.bundleDirectory = bundleDirectory
        self.modelConfiguration = configuration
        let manifestURL = bundleDirectory.appendingPathComponent("manifest.json")
        let manifestData = try GPTSoVITSRuntimeProfiler.measure("bundle_driver.manifest.read") {
            try Data(contentsOf: manifestURL)
        }
        self.manifest = try GPTSoVITSRuntimeProfiler.measure("bundle_driver.manifest.decode") {
            try JSONDecoder().decode(
                GPTSoVITSChineseSynthesisBundleManifest.self,
                from: manifestData
            )
        }
        let manifest = self.manifest

        let resolvedArtifacts = GPTSoVITSRuntimeProfiler.measure("bundle_driver.artifacts.resolve") {
            (
                t2sBundleURL: Self.resolveArtifactURL(manifest.artifacts.t2sBundle, relativeTo: bundleDirectory),
                vitsBundleURL: Self.resolveArtifactURL(manifest.artifacts.vitsBundle, relativeTo: bundleDirectory),
                zhBertURL: Self.resolveArtifactURL(manifest.artifacts.zhBertChar, relativeTo: bundleDirectory),
                tokenizerURL: Self.resolveArtifactURL(manifest.artifacts.tokenizer, relativeTo: bundleDirectory),
                g2pwBundleURL: manifest.artifacts.g2pwBundle.map {
                    Self.resolveArtifactURL($0, relativeTo: bundleDirectory)
                },
                englishFrontendBundleDirectory: manifest.artifacts.englishFrontendBundle.map {
                    Self.resolveArtifactURL($0, relativeTo: bundleDirectory)
                },
                yueFrontendBundleDirectory: manifest.artifacts.yueFrontendBundle.map {
                    Self.resolveArtifactURL($0, relativeTo: bundleDirectory)
                },
                japaneseFrontendBundleDirectory: manifest.artifacts.japaneseFrontendBundle.map {
                    Self.resolveArtifactURL($0, relativeTo: bundleDirectory)
                },
                koreanFrontendBundleDirectory: manifest.artifacts.koreanFrontendBundle.map {
                    Self.resolveArtifactURL($0, relativeTo: bundleDirectory)
                },
                promptBundleURL: manifest.artifacts.promptBundle.map {
                    Self.resolvePromptBundleURL($0, relativeTo: bundleDirectory)
                }
            )
        }
        self.englishFrontendBundleDirectory = resolvedArtifacts.englishFrontendBundleDirectory
        self.yueFrontendBundleDirectory = resolvedArtifacts.yueFrontendBundleDirectory
        self.japaneseFrontendBundleDirectory = resolvedArtifacts.japaneseFrontendBundleDirectory
        self.koreanFrontendBundleDirectory = resolvedArtifacts.koreanFrontendBundleDirectory
        self.promptBundleDirectory = resolvedArtifacts.promptBundleURL

        if let promptBundleURL = resolvedArtifacts.promptBundleURL {
            self.pipeline = try GPTSoVITSRuntimeProfiler.measure("bundle_driver.pipeline.init") {
                try GPTSoVITSChineseSynthesisPipeline(
                    t2sBundleDirectory: resolvedArtifacts.t2sBundleURL,
                    zhBertModelURL: resolvedArtifacts.zhBertURL,
                    tokenizerJSONURL: resolvedArtifacts.tokenizerURL,
                    promptBundleDirectory: promptBundleURL,
                    vitsBundleDirectory: resolvedArtifacts.vitsBundleURL,
                    g2pwBundleDirectory: resolvedArtifacts.g2pwBundleURL,
                    configuration: configuration
                )
            }
        } else {
            self.pipeline = try GPTSoVITSRuntimeProfiler.measure("bundle_driver.pipeline.init") {
                try GPTSoVITSChineseSynthesisPipeline(
                    t2sBundleDirectory: resolvedArtifacts.t2sBundleURL,
                    zhBertModelURL: resolvedArtifacts.zhBertURL,
                    tokenizerJSONURL: resolvedArtifacts.tokenizerURL,
                    vitsBundleDirectory: resolvedArtifacts.vitsBundleURL,
                    g2pwBundleDirectory: resolvedArtifacts.g2pwBundleURL,
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
        try pipeline.resolveDefaultTextPhoneBackend(
            for: language,
            englishFrontendBundleDirectory: englishFrontendBundleDirectory,
            yueFrontendBundleDirectory: yueFrontendBundleDirectory,
            japaneseFrontendBundleDirectory: japaneseFrontendBundleDirectory,
            koreanFrontendBundleDirectory: koreanFrontendBundleDirectory
        )
        return try pipeline.synthesize(
            prompts: prompts,
            promptText: promptText,
            targetText: targetText,
            referenceAudio: referenceAudio,
            language: language,
            splitMethod: splitMethod,
            noiseScale: noiseScale,
            seed: seed
        )
    }

    public func synthesizeCrossLingual(
        prompts: [Int32],
        promptText: String,
        targetText: String,
        promptLanguage: GPTSoVITSTextLanguage,
        targetLanguage: GPTSoVITSTextLanguage,
        referenceAudio: ReferenceAudioSamples,
        splitMethod: GPTSoVITSTextSplitMethod = .cut5,
        noiseScale: Float? = nil,
        seed: UInt64? = nil
    ) throws -> GPTSoVITSChineseSynthesisResult {
        try resolveTextPhoneBackend(for: promptLanguage)
        let preparedPrompt = try pipeline.textPreparer.preparePromptText(promptText, language: promptLanguage)
        try resolveTextPhoneBackend(for: targetLanguage)
        let preparedTargets = try pipeline.textPreparer.prepareTargetSegments(
            text: targetText,
            language: targetLanguage,
            splitMethod: splitMethod
        )
        let conditioning = try pipeline.speakerConditioned.vits.prepareReferenceConditioning(referenceAudio: referenceAudio)
        return try synthesizePreparedCrossLingual(
            prompts: prompts,
            promptLength: prompts.count,
            preparedPrompt: preparedPrompt,
            preparedTargets: preparedTargets,
            conditioning: conditioning,
            noiseScale: noiseScale,
            seed: seed
        )
    }

    public func synthesize(
        prompts: [Int32],
        promptText: String,
        targetText: String,
        referenceAudioChannels: [[Float]],
        referenceAudioSampleRate: Int,
        language: GPTSoVITSTextLanguage = .zh,
        splitMethod: GPTSoVITSTextSplitMethod = .cut5,
        noiseScale: Float? = nil,
        seed: UInt64? = nil
    ) throws -> GPTSoVITSChineseSynthesisResult {
        try pipeline.resolveDefaultTextPhoneBackend(
            for: language,
            englishFrontendBundleDirectory: englishFrontendBundleDirectory,
            yueFrontendBundleDirectory: yueFrontendBundleDirectory,
            japaneseFrontendBundleDirectory: japaneseFrontendBundleDirectory,
            koreanFrontendBundleDirectory: koreanFrontendBundleDirectory
        )
        return try pipeline.synthesize(
            prompts: prompts,
            promptText: promptText,
            targetText: targetText,
            referenceAudioChannels: referenceAudioChannels,
            referenceAudioSampleRate: referenceAudioSampleRate,
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
        try pipeline.resolveDefaultTextPhoneBackend(
            for: language,
            englishFrontendBundleDirectory: englishFrontendBundleDirectory,
            yueFrontendBundleDirectory: yueFrontendBundleDirectory,
            japaneseFrontendBundleDirectory: japaneseFrontendBundleDirectory,
            koreanFrontendBundleDirectory: koreanFrontendBundleDirectory
        )
        return try pipeline.synthesize(
            promptText: promptText,
            targetText: targetText,
            referenceAudio: referenceAudio,
            language: language,
            splitMethod: splitMethod,
            noiseScale: noiseScale,
            seed: seed
        )
    }

    public func synthesizeCrossLingual(
        promptText: String,
        targetText: String,
        promptLanguage: GPTSoVITSTextLanguage,
        targetLanguage: GPTSoVITSTextLanguage,
        referenceAudio: ReferenceAudioSamples,
        splitMethod: GPTSoVITSTextSplitMethod = .cut5,
        noiseScale: Float? = nil,
        seed: UInt64? = nil
    ) throws -> GPTSoVITSChineseSynthesisResult {
        try synthesizeCrossLingual(
            promptText: promptText,
            targetText: targetText,
            promptLanguage: promptLanguage,
            targetLanguage: targetLanguage,
            promptReferenceAudio: referenceAudio,
            conditioningReferenceAudio: referenceAudio,
            splitMethod: splitMethod,
            noiseScale: noiseScale,
            seed: seed
        )
    }

    public func synthesize(
        promptText: String,
        targetText: String,
        referenceAudioChannels: [[Float]],
        referenceAudioSampleRate: Int,
        language: GPTSoVITSTextLanguage = .zh,
        splitMethod: GPTSoVITSTextSplitMethod = .cut5,
        noiseScale: Float? = nil,
        seed: UInt64? = nil
    ) throws -> GPTSoVITSChineseSynthesisResult {
        try pipeline.resolveDefaultTextPhoneBackend(
            for: language,
            englishFrontendBundleDirectory: englishFrontendBundleDirectory,
            yueFrontendBundleDirectory: yueFrontendBundleDirectory,
            japaneseFrontendBundleDirectory: japaneseFrontendBundleDirectory,
            koreanFrontendBundleDirectory: koreanFrontendBundleDirectory
        )
        return try pipeline.synthesize(
            promptText: promptText,
            targetText: targetText,
            referenceAudioChannels: referenceAudioChannels,
            referenceAudioSampleRate: referenceAudioSampleRate,
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
        try pipeline.resolveDefaultTextPhoneBackend(
            for: language,
            englishFrontendBundleDirectory: englishFrontendBundleDirectory,
            yueFrontendBundleDirectory: yueFrontendBundleDirectory,
            japaneseFrontendBundleDirectory: japaneseFrontendBundleDirectory,
            koreanFrontendBundleDirectory: koreanFrontendBundleDirectory
        )
        return try pipeline.synthesize(
            promptText: promptText,
            targetText: targetText,
            promptReferenceAudio: promptReferenceAudio,
            conditioningReferenceAudio: conditioningReferenceAudio,
            language: language,
            splitMethod: splitMethod,
            noiseScale: noiseScale,
            seed: seed
        )
    }

    public func synthesizeCrossLingual(
        promptText: String,
        targetText: String,
        promptLanguage: GPTSoVITSTextLanguage,
        targetLanguage: GPTSoVITSTextLanguage,
        promptReferenceAudio: ReferenceAudioSamples,
        conditioningReferenceAudio: ReferenceAudioSamples,
        splitMethod: GPTSoVITSTextSplitMethod = .cut5,
        noiseScale: Float? = nil,
        seed: UInt64? = nil
    ) throws -> GPTSoVITSChineseSynthesisResult {
        let promptSemantic = try makePromptSemanticPipeline()
        let promptResult = try promptSemantic.extractPrompts(referenceAudio: promptReferenceAudio)
        try resolveTextPhoneBackend(for: promptLanguage)
        let preparedPrompt = try pipeline.textPreparer.preparePromptText(promptText, language: promptLanguage)
        try resolveTextPhoneBackend(for: targetLanguage)
        let preparedTargets = try pipeline.textPreparer.prepareTargetSegments(
            text: targetText,
            language: targetLanguage,
            splitMethod: splitMethod
        )
        let conditioning = try pipeline.speakerConditioned.vits.prepareReferenceConditioning(
            referenceAudio: conditioningReferenceAudio
        )
        let synthesized = try synthesizePreparedCrossLingual(
            prompts: promptResult.prompts,
            promptLength: promptResult.promptCount,
            preparedPrompt: preparedPrompt,
            preparedTargets: preparedTargets,
            conditioning: conditioning,
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

    private func resolveTextPhoneBackend(
        for language: GPTSoVITSTextLanguage
    ) throws {
        try pipeline.resolveDefaultTextPhoneBackend(
            for: language,
            englishFrontendBundleDirectory: englishFrontendBundleDirectory,
            yueFrontendBundleDirectory: yueFrontendBundleDirectory,
            japaneseFrontendBundleDirectory: japaneseFrontendBundleDirectory,
            koreanFrontendBundleDirectory: koreanFrontendBundleDirectory
        )
    }

    private func makePromptSemanticPipeline() throws -> GPTSoVITSPromptSemanticPipeline {
        if let promptSemantic = pipeline.promptSemantic {
            return promptSemantic
        }
        guard let promptBundleDirectory else {
            throw GPTSoVITSChineseSynthesisPipelineError.promptConditioningUnavailable
        }
        return try GPTSoVITSPromptSemanticPipeline(
            bundleDirectory: promptBundleDirectory,
            configuration: modelConfiguration
        )
    }

    private func synthesizePreparedCrossLingual(
        prompts: [Int32],
        promptLength: Int,
        preparedPrompt: GPTSoVITST2SPreparedTextInput,
        preparedTargets: [GPTSoVITST2SPreparedSegment],
        conditioning: PreparedReferenceConditioning,
        noiseScale: Float?,
        seed: UInt64?
    ) throws -> GPTSoVITSChineseSynthesisResult {
        let promptsArray = try pipeline.speakerConditioned.t2s.makeInt32Array(
            shape: [1, max(prompts.count, 1)],
            values: prompts + Array(repeating: 0, count: max(0, max(prompts.count, 1) - prompts.count))
        )
        let usesDynamicPrompts = pipeline.speakerConditioned.t2s.manifest.runtime.shapes?.promptLenRange != nil
        let resolvedPromptsArray = usesDynamicPrompts
            ? promptsArray
            : try pipeline.speakerConditioned.t2s.makeInt32Array(
                shape: pipeline.speakerConditioned.t2s.prefillModel.modelDescription
                    .inputDescriptionsByName["prompts"]?
                    .multiArrayConstraint?
                    .shape
                    .map { Int(truncating: $0) } ?? [1, prompts.count],
                values: prompts + Array(
                    repeating: 0,
                    count: max(
                        0,
                        (pipeline.speakerConditioned.t2s.prefillModel.modelDescription
                            .inputDescriptionsByName["prompts"]?
                            .multiArrayConstraint?
                            .shape
                            .last
                            .map { Int(truncating: $0) } ?? prompts.count) - prompts.count
                    )
                )
            )
        let t2sSamplingRNGBox = seed.map(T2SSamplingRNGBox.init(seed:))
        let synthesizedSegments = try preparedTargets.map { segment in
            let vitsText = try pipeline.speakerConditioned.makePaddedVITSText(phoneIDs: segment.input.phoneIDs)
            return GPTSoVITSChineseSynthesisSegmentResult(
                preparedSegment: segment,
                synthesis: try pipeline.speakerConditioned.synthesize(
                    prompts: resolvedPromptsArray,
                    promptLength: promptLength,
                    refSeq: preparedPrompt.seq,
                    refSeqLength: preparedPrompt.phoneCount,
                    textSeq: segment.input.seq,
                    textSeqLength: segment.input.phoneCount,
                    refBert: preparedPrompt.bert,
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
            prompt: preparedPrompt,
            segments: synthesizedSegments
        )
    }

    private static func resolveArtifactURL(
        _ artifact: GPTSoVITSChineseSynthesisBundleManifest.Artifact,
        relativeTo bundleDirectory: URL
    ) -> URL {
        if !artifact.path.isEmpty {
            let explicitURL = resolvedURL(for: artifact.path, relativeTo: bundleDirectory)
            if FileManager.default.fileExists(atPath: explicitURL.path) {
                return explicitURL
            }
        }
        return resolvedURL(for: artifact.filename, relativeTo: bundleDirectory)
    }

    private static func resolvedURL(for rawPath: String, relativeTo bundleDirectory: URL) -> URL {
        if (rawPath as NSString).isAbsolutePath {
            return URL(fileURLWithPath: rawPath, isDirectory: false)
        }
        return bundleDirectory.appendingPathComponent(rawPath)
    }

    private static func resolvePromptBundleURL(
        _ artifact: GPTSoVITSChineseSynthesisBundleManifest.Artifact,
        relativeTo bundleDirectory: URL
    ) -> URL {
        let resolved = resolveArtifactURL(artifact, relativeTo: bundleDirectory)
        let basename = resolved.lastPathComponent
        if basename.hasSuffix("_dynamic") || basename.hasSuffix("_dynamic_cpuonly") {
            return resolved
        }

        let fileManager = FileManager.default
        let override = ProcessInfo.processInfo.environment["GPTSOVITS_PROMPT_BUNDLE_VARIANT"]?.lowercased()
        let computeUnits = ProcessInfo.processInfo.environment["GPTSOVITS_COREML_COMPUTE_UNITS"]?.lowercased()
        let preferredPaths: [String]
        switch override {
        case "legacy":
            preferredPaths = [resolved.path]
        case "dynamic":
            preferredPaths = [resolved.path + "_dynamic", resolved.path]
        case "dynamic_cpuonly":
            preferredPaths = [resolved.path + "_dynamic_cpuonly", resolved.path + "_dynamic", resolved.path]
        default:
            if computeUnits == "cpu_only" {
                preferredPaths = [resolved.path + "_dynamic_cpuonly", resolved.path + "_dynamic", resolved.path]
            } else {
                preferredPaths = [resolved.path + "_dynamic", resolved.path + "_dynamic_cpuonly", resolved.path]
            }
        }

        for path in preferredPaths where fileManager.fileExists(atPath: path) {
            return URL(fileURLWithPath: path, isDirectory: true)
        }
        return resolved
    }
}
