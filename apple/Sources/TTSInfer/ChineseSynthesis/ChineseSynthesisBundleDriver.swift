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
    public let englishFrontendBundleDirectory: URL?
    public let yueFrontendBundleDirectory: URL?
    public let japaneseFrontendBundleDirectory: URL?
    public let koreanFrontendBundleDirectory: URL?

    public init(bundleDirectory: URL, configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        self.bundleDirectory = bundleDirectory
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
