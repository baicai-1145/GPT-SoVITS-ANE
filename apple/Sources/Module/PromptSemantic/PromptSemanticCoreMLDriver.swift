import CoreML
import Foundation

public struct PromptSemanticBundleManifest: Decodable {
    public struct Artifact: Decodable {
        public let target: String
        public let filename: String
        public let path: String
    }

    public struct Artifacts: Decodable {
        public let cnhubertEncoder: Artifact
        public let sslLatentExtractor: Artifact

        private enum CodingKeys: String, CodingKey {
            case cnhubertEncoder = "cnhubert_encoder"
            case sslLatentExtractor = "ssl_latent_extractor"
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

            public let inputSampleCount: Int
            public let sslContentShape: [Int]
            public let promptShape: [Int]
            public let promptLen: Int
            public let promptHopSamples: Int
            public let inputSampleCountRange: ShapeRange?
            public let sslFrameRange: ShapeRange?
            public let promptLenRange: ShapeRange?

            private enum CodingKeys: String, CodingKey {
                case inputSampleCount = "input_sample_count"
                case sslContentShape = "ssl_content_shape"
                case promptShape = "prompt_shape"
                case promptLen = "prompt_len"
                case promptHopSamples = "prompt_hop_samples"
                case inputSampleCountRange = "input_sample_count_range"
                case sslFrameRange = "ssl_frame_range"
                case promptLenRange = "prompt_len_range"
            }
        }

        public struct AudioInputContract: Decodable {
            public struct Normalization: Decodable {
                public let source: String
                public let doNormalize: Bool
                public let formula: String

                private enum CodingKeys: String, CodingKey {
                    case source
                    case doNormalize = "do_normalize"
                    case formula
                }
            }

            public struct Padding: Decodable {
                public let mode: String
                public let targetSampleCount: Int

                private enum CodingKeys: String, CodingKey {
                    case mode
                    case targetSampleCount = "target_sample_count"
                }
            }

            public let channelPolicy: String
            public let targetSampleRate: Int
            public let trailingSilenceSampleCount: Int
            public let normalization: Normalization
            public let padding: Padding

            private enum CodingKeys: String, CodingKey {
                case channelPolicy = "channel_policy"
                case targetSampleRate = "target_sample_rate"
                case trailingSilenceSampleCount = "trailing_silence_sample_count"
                case normalization
                case padding
            }
        }

        public let shapes: Shapes
        public let audioInputContract: AudioInputContract

        private enum CodingKeys: String, CodingKey {
            case shapes
            case audioInputContract = "audio_input_contract"
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

public final class PromptSemanticCoreMLDriver {
    public let manifest: PromptSemanticBundleManifest
    public let cnhubert: CNHubertCoreMLDriver
    public let sslLatentExtractor: SSLLatentExtractorCoreMLDriver

    public init(bundleDirectory: URL, configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        let manifestURL = bundleDirectory.appendingPathComponent("manifest.json")
        let manifestData = try Data(contentsOf: manifestURL)
        let decoder = JSONDecoder()
        self.manifest = try decoder.decode(PromptSemanticBundleManifest.self, from: manifestData)

        let cnhubertURL = bundleDirectory.appendingPathComponent(manifest.artifacts.cnhubertEncoder.filename)
        let sslLatentURL = bundleDirectory.appendingPathComponent(manifest.artifacts.sslLatentExtractor.filename)
        self.cnhubert = try CNHubertCoreMLDriver(modelURL: cnhubertURL, configuration: configuration)
        self.sslLatentExtractor = try SSLLatentExtractorCoreMLDriver(modelURL: sslLatentURL, configuration: configuration)
    }
}
