import Foundation

public struct JapaneseFrontendBundleManifest: Decodable {
    public struct Artifact: Decodable {
        public let target: String
        public let filename: String
        public let path: String
    }

    public struct Artifacts: Decodable {
        public let runtimeAssets: Artifact
        public let openjtalkDictionary: Artifact
        public let jaUserDictionary: Artifact
        public let openjtalkDynamicLibrary: Artifact

        private enum CodingKeys: String, CodingKey {
            case runtimeAssets = "runtime_assets"
            case openjtalkDictionary = "openjtalk_dictionary"
            case jaUserDictionary = "ja_user_dictionary"
            case openjtalkDynamicLibrary = "openjtalk_dynamic_library"
        }
    }

    public struct Runtime: Decodable {
        public struct Features: Decodable {
            public let openjtalkDictionaryStaged: Bool
            public let jaUserDictionaryStaged: Bool
            public let openjtalkDynamicLibraryStaged: Bool?

            private enum CodingKeys: String, CodingKey {
                case openjtalkDictionaryStaged = "openjtalk_dictionary_staged"
                case jaUserDictionaryStaged = "ja_user_dictionary_staged"
                case openjtalkDynamicLibraryStaged = "openjtalk_dynamic_library_staged"
            }
        }

        public struct Materialization: Decodable {
            public let openjtalkDictionaryMode: String
            public let jaUserDictionaryMode: String
            public let openjtalkDynamicLibraryMode: String?

            private enum CodingKeys: String, CodingKey {
                case openjtalkDictionaryMode = "openjtalk_dictionary_mode"
                case jaUserDictionaryMode = "ja_user_dictionary_mode"
                case openjtalkDynamicLibraryMode = "openjtalk_dynamic_library_mode"
            }
        }

        public let features: Features
        public let materialization: Materialization
        public let nextStep: String

        private enum CodingKeys: String, CodingKey {
            case features
            case materialization
            case nextStep = "next_step"
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

public struct JapaneseFrontendRuntimeAssets: Decodable {
    public struct RegexReplacement: Decodable {
        public let pattern: String
        public let replacement: String
    }

    public struct ProjectAssets: Decodable {
        public struct RuntimeContract: Decodable {
            public let sentenceFlow: [String]
            public let withProsody: Bool

            private enum CodingKeys: String, CodingKey {
                case sentenceFlow = "sentence_flow"
                case withProsody = "with_prosody"
            }
        }

        public let japaneseCharactersPattern: String
        public let japaneseMarksPattern: String
        public let symbolsToJapanese: [RegexReplacement]
        public let realSokuon: [RegexReplacement]
        public let realHatsuon: [RegexReplacement]
        public let prosodyMarks: [String]
        public let postReplaceMap: [String: String]
        public let runtimeContract: RuntimeContract

        private enum CodingKeys: String, CodingKey {
            case japaneseCharactersPattern = "japanese_characters_pattern"
            case japaneseMarksPattern = "japanese_marks_pattern"
            case symbolsToJapanese = "symbols_to_japanese"
            case realSokuon = "real_sokuon"
            case realHatsuon = "real_hatsuon"
            case prosodyMarks = "prosody_marks"
            case postReplaceMap = "post_replace_map"
            case runtimeContract = "runtime_contract"
        }
    }

    public struct PyopenjtalkProbe: Decodable {
        public struct Sample: Decodable {
            public let text: String
            public let g2p: String
            public let frontendSize: Int

            private enum CodingKeys: String, CodingKey {
                case text
                case g2p
                case frontendSize = "frontend_size"
            }
        }

        public let dictionaryDir: String
        public let samples: [Sample]

        private enum CodingKeys: String, CodingKey {
            case dictionaryDir = "dictionary_dir"
            case samples
        }
    }

    public struct ProjectProbe: Decodable {
        public struct Sample: Decodable {
            public struct PhoneUnit: Decodable {
                public let unitType: String
                public let text: String
                public let normText: String
                public let phones: [String]
                public let charStart: Int?
                public let charEnd: Int?
                public let phoneStart: Int?
                public let phoneEnd: Int?
                public let phoneCount: Int?

                private enum CodingKeys: String, CodingKey {
                    case unitType = "unit_type"
                    case text
                    case normText = "norm_text"
                    case phones
                    case charStart = "char_start"
                    case charEnd = "char_end"
                    case phoneStart = "phone_start"
                    case phoneEnd = "phone_end"
                    case phoneCount = "phone_count"
                }
            }

            public let text: String
            public let normalized: String
            public let phones: [String]
            public let phoneUnits: [PhoneUnit]

            private enum CodingKeys: String, CodingKey {
                case text
                case normalized
                case phones
                case phoneUnits = "phone_units"
            }
        }

        public let samples: [Sample]
    }

    public let formatVersion: Int
    public let projectAssets: ProjectAssets
    public let pyopenjtalkProbe: PyopenjtalkProbe
    public let projectProbe: ProjectProbe
    public let blockers: [String]

    private enum CodingKeys: String, CodingKey {
        case formatVersion = "format_version"
        case projectAssets = "project_assets"
        case pyopenjtalkProbe = "pyopenjtalk_probe"
        case projectProbe = "project_probe"
        case blockers
    }
}

public enum JapaneseFrontendBundleError: LocalizedError {
    case invalidBundleType(String)

    public var errorDescription: String? {
        switch self {
        case let .invalidBundleType(bundleType):
            return "日语 frontend bundle_type 不匹配: \(bundleType)"
        }
    }
}

public final class JapaneseFrontendBundle {
    public let manifest: JapaneseFrontendBundleManifest
    public let runtimeAssets: JapaneseFrontendRuntimeAssets
    public let bundleDirectory: URL
    public let openjtalkDictionaryDirectory: URL
    public let jaUserDictionaryDirectory: URL
    public let openjtalkDynamicLibraryURL: URL

    public static func bundleType(at bundleDirectory: URL) -> String? {
        let manifestURL = bundleDirectory.appendingPathComponent("manifest.json")
        guard let data = try? Data(contentsOf: manifestURL) else {
            return nil
        }
        return (try? JSONDecoder().decode(JapaneseFrontendBundleManifest.self, from: data))?.bundleType
    }

    public static func isBundleDirectory(_ bundleDirectory: URL) -> Bool {
        bundleType(at: bundleDirectory) == "gpt_sovits_japanese_frontend_bundle"
    }

    public init(bundleDirectory: URL) throws {
        self.bundleDirectory = bundleDirectory
        let manifestURL = bundleDirectory.appendingPathComponent("manifest.json")
        let manifestData = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(JapaneseFrontendBundleManifest.self, from: manifestData)
        guard manifest.bundleType == "gpt_sovits_japanese_frontend_bundle" else {
            throw JapaneseFrontendBundleError.invalidBundleType(manifest.bundleType)
        }

        let assetsURL = Self.resolveArtifactURL(manifest.artifacts.runtimeAssets, relativeTo: bundleDirectory)
        self.openjtalkDictionaryDirectory = Self.resolveArtifactURL(manifest.artifacts.openjtalkDictionary, relativeTo: bundleDirectory)
        self.jaUserDictionaryDirectory = Self.resolveArtifactURL(manifest.artifacts.jaUserDictionary, relativeTo: bundleDirectory)
        self.openjtalkDynamicLibraryURL = Self.resolveArtifactURL(
            manifest.artifacts.openjtalkDynamicLibrary,
            relativeTo: bundleDirectory
        )
        self.manifest = manifest
        self.runtimeAssets = try JSONDecoder().decode(
            JapaneseFrontendRuntimeAssets.self,
            from: Data(contentsOf: assetsURL)
        )
    }

    private static func resolveArtifactURL(
        _ artifact: JapaneseFrontendBundleManifest.Artifact,
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
        let pathExtension = URL(fileURLWithPath: rawPath).pathExtension.lowercased()
        let isDirectory = pathExtension.isEmpty
        if (rawPath as NSString).isAbsolutePath {
            return URL(fileURLWithPath: rawPath, isDirectory: isDirectory)
        }
        return bundleDirectory.appendingPathComponent(rawPath, isDirectory: isDirectory)
    }
}
