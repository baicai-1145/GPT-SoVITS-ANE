import Foundation

public struct KoreanFrontendBundleManifest: Decodable {
    public struct Artifact: Decodable {
        public let target: String
        public let filename: String
        public let path: String
    }

    public struct Artifacts: Decodable {
        public let runtimeAssets: Artifact
        public let mecabDictionary: Artifact
        public let mecabRCFile: Artifact
        public let mecabDynamicLibrary: Artifact

        private enum CodingKeys: String, CodingKey {
            case runtimeAssets = "runtime_assets"
            case mecabDictionary = "mecab_dictionary"
            case mecabRCFile = "mecab_rc_file"
            case mecabDynamicLibrary = "mecab_dynamic_library"
        }
    }

    public struct Runtime: Decodable {
        public struct Features: Decodable {
            public let g2pk2StaticRules: Bool
            public let g2pk2SpecialRules: Bool?
            public let g2pk2EnglishAssets: Bool?
            public let g2pk2NumeralsAssets: Bool?
            public let koPronStaticTables: Bool
            public let mecabDictionaryStaged: Bool
            public let mecabRCFileStaged: Bool?
            public let mecabDynamicLibraryStaged: Bool?

            private enum CodingKeys: String, CodingKey {
                case g2pk2StaticRules = "g2pk2_static_rules"
                case g2pk2SpecialRules = "g2pk2_special_rules"
                case g2pk2EnglishAssets = "g2pk2_english_assets"
                case g2pk2NumeralsAssets = "g2pk2_numerals_assets"
                case koPronStaticTables = "ko_pron_static_tables"
                case mecabDictionaryStaged = "mecab_dictionary_staged"
                case mecabRCFileStaged = "mecab_rc_file_staged"
                case mecabDynamicLibraryStaged = "mecab_dynamic_library_staged"
            }
        }

        public struct Materialization: Decodable {
            public let mecabDictionaryMode: String
            public let mecabRCFileMode: String?
            public let mecabDynamicLibraryMode: String?

            private enum CodingKeys: String, CodingKey {
                case mecabDictionaryMode = "mecab_dictionary_mode"
                case mecabRCFileMode = "mecab_rc_file_mode"
                case mecabDynamicLibraryMode = "mecab_dynamic_library_mode"
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

public struct KoreanFrontendRuntimeAssets: Decodable {
    public struct RegexReplacement: Decodable {
        public let pattern: String
        public let replacement: String
    }

    public struct ProjectAssets: Decodable {
        public let koreanClassifiers: [String]
        public let hangulDivided: [RegexReplacement]
        public let latinToHangul: [RegexReplacement]
        public let ipaToLazyIpa: [RegexReplacement]
        public let postReplaceMap: [String: String]
        public let separatorContract: [String: String]
        public let runtimeContract: RuntimeContract

        public struct RuntimeContract: Decodable {
            public let transformSteps: [String]
            public let romanizationSteps: [String]

            private enum CodingKeys: String, CodingKey {
                case transformSteps = "transform_steps"
                case romanizationSteps = "romanization_steps"
            }
        }

        private enum CodingKeys: String, CodingKey {
            case koreanClassifiers = "korean_classifiers"
            case hangulDivided = "hangul_divided"
            case latinToHangul = "latin_to_hangul"
            case ipaToLazyIpa = "ipa_to_lazy_ipa"
            case postReplaceMap = "post_replace_map"
            case separatorContract = "separator_contract"
            case runtimeContract = "runtime_contract"
        }
    }

    public struct G2pk2Assets: Decodable {
        public struct TableRule: Decodable {
            public let pattern: String
            public let replacement: String
            public let ruleIDs: [String]

            private enum CodingKeys: String, CodingKey {
                case pattern
                case replacement
                case ruleIDs = "rule_ids"
            }
        }

        public let packageRoot: String
        public let table: [TableRule]
        public let ruleIDToText: [String: String]
        public let idiomsLines: [String]
        public let rulesText: String

        private enum CodingKeys: String, CodingKey {
            case packageRoot = "package_root"
            case table
            case ruleIDToText = "rule_id_to_text"
            case idiomsLines = "idioms_lines"
            case rulesText = "rules_text"
        }
    }

    public struct G2pk2EffectiveSpecialAssets: Decodable {
        public struct Step: Decodable {
            public let name: String
            public let ruleID: String
            public let replacements: [RegexReplacement]

            private enum CodingKeys: String, CodingKey {
                case name
                case ruleID = "rule_id"
                case replacements
            }
        }

        public let steps: [Step]
    }

    public struct G2pk2EnglishAssets: Decodable {
        public let adjustReplacements: [RegexReplacement]
        public let toChoseong: [String: String]
        public let toJungseong: [String: String]
        public let toJongseong: [String: String]
        public let reconstructPairs: [RegexReplacement]
        public let shortVowels: [String]
        public let vowels: [String]
        public let consonants: [String]
        public let syllableFinalOrConsonants: [String]
        public let cmuDict: [String: [[String]]]

        private enum CodingKeys: String, CodingKey {
            case adjustReplacements = "adjust_replacements"
            case toChoseong = "to_choseong"
            case toJungseong = "to_jungseong"
            case toJongseong = "to_jongseong"
            case reconstructPairs = "reconstruct_pairs"
            case shortVowels = "short_vowels"
            case vowels
            case consonants
            case syllableFinalOrConsonants = "syllable_final_or_consonants"
            case cmuDict = "cmu_dict"
        }
    }

    public struct G2pk2NumeralsAssets: Decodable {
        public let boundNouns: [String]
        public let digits: [String]
        public let digitNames: [String]
        public let nonZeroDigits: [String]
        public let nonZeroDigitNames: [String]
        public let modifiers: [String]
        public let decimals: [String]

        private enum CodingKeys: String, CodingKey {
            case boundNouns = "bound_nouns"
            case digits
            case digitNames = "digit_names"
            case nonZeroDigits = "non_zero_digits"
            case nonZeroDigitNames = "non_zero_digit_names"
            case modifiers
            case decimals
        }
    }

    public struct KoPronAssets: Decodable {
        public let packageRoot: String
        public let vowels: [String: [String]]
        public let boundary: [String: [String]]

        private enum CodingKeys: String, CodingKey {
            case packageRoot = "package_root"
            case vowels
            case boundary
        }
    }

    public struct MecabProbe: Decodable {
        public struct Sample: Decodable {
            public struct Token: Decodable {
                public let surface: String
                public let tag: String
            }

            public let text: String
            public let pos: [Token]
        }

        public let taggerClass: String
        public let samples: [Sample]

        private enum CodingKeys: String, CodingKey {
            case taggerClass = "tagger_class"
            case samples
        }
    }

    public let formatVersion: Int
    public let projectAssets: ProjectAssets
    public let g2pk2Assets: G2pk2Assets
    public let g2pk2EffectiveSpecialAssets: G2pk2EffectiveSpecialAssets
    public let g2pk2EnglishAssets: G2pk2EnglishAssets
    public let g2pk2NumeralsAssets: G2pk2NumeralsAssets
    public let koPronAssets: KoPronAssets
    public let mecabProbe: MecabProbe
    public let blockers: [String]

    private enum CodingKeys: String, CodingKey {
        case formatVersion = "format_version"
        case projectAssets = "project_assets"
        case g2pk2Assets = "g2pk2_assets"
        case g2pk2EffectiveSpecialAssets = "g2pk2_effective_special_assets"
        case g2pk2EnglishAssets = "g2pk2_english_assets"
        case g2pk2NumeralsAssets = "g2pk2_numerals_assets"
        case koPronAssets = "ko_pron_assets"
        case mecabProbe = "mecab_probe"
        case blockers
    }
}

public enum KoreanFrontendBundleError: LocalizedError {
    case invalidBundleType(String)

    public var errorDescription: String? {
        switch self {
        case let .invalidBundleType(bundleType):
            return "韩语 frontend bundle_type 不匹配: \(bundleType)"
        }
    }
}

public final class KoreanFrontendBundle {
    public let manifest: KoreanFrontendBundleManifest
    public let runtimeAssets: KoreanFrontendRuntimeAssets
    public let bundleDirectory: URL
    public let mecabDictionaryDirectory: URL
    public let mecabRCFileURL: URL
    public let mecabDynamicLibraryURL: URL

    public static func bundleType(at bundleDirectory: URL) -> String? {
        let manifestURL = bundleDirectory.appendingPathComponent("manifest.json")
        guard let data = try? Data(contentsOf: manifestURL) else {
            return nil
        }
        return (try? JSONDecoder().decode(KoreanFrontendBundleManifest.self, from: data))?.bundleType
    }

    public static func isBundleDirectory(_ bundleDirectory: URL) -> Bool {
        bundleType(at: bundleDirectory) == "gpt_sovits_korean_frontend_bundle"
    }

    public init(bundleDirectory: URL) throws {
        self.bundleDirectory = bundleDirectory
        let manifestURL = bundleDirectory.appendingPathComponent("manifest.json")
        let manifestData = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(KoreanFrontendBundleManifest.self, from: manifestData)
        guard manifest.bundleType == "gpt_sovits_korean_frontend_bundle" else {
            throw KoreanFrontendBundleError.invalidBundleType(manifest.bundleType)
        }

        let assetsURL = Self.resolveArtifactURL(manifest.artifacts.runtimeAssets, relativeTo: bundleDirectory)
        self.mecabDictionaryDirectory = Self.resolveArtifactURL(manifest.artifacts.mecabDictionary, relativeTo: bundleDirectory)
        self.mecabRCFileURL = Self.resolveArtifactURL(manifest.artifacts.mecabRCFile, relativeTo: bundleDirectory)
        self.mecabDynamicLibraryURL = Self.resolveArtifactURL(manifest.artifacts.mecabDynamicLibrary, relativeTo: bundleDirectory)
        self.manifest = manifest
        self.runtimeAssets = try JSONDecoder().decode(
            KoreanFrontendRuntimeAssets.self,
            from: Data(contentsOf: assetsURL)
        )
    }

    private static func resolveArtifactURL(
        _ artifact: KoreanFrontendBundleManifest.Artifact,
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
