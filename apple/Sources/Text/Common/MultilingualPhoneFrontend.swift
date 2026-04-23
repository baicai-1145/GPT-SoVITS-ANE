import Foundation

public enum GPTSoVITSTextPhoneUnitType: String {
    case word
    case char
    case punct
    case prosody
    case wordGroup = "word_group"
    case space
}

public struct GPTSoVITSTextPhoneUnit {
    public let unitType: GPTSoVITSTextPhoneUnitType
    public let text: String
    public let normText: String
    public let pos: String?
    public let phones: [String]
    public let phoneIDs: [Int]
    public let charStart: Int
    public let charEnd: Int
    public let phoneStart: Int
    public let phoneEnd: Int
    public let phoneCount: Int
}

public struct GPTSoVITSTextPhoneResult {
    public let sourceText: String
    public let normalizedText: String
    public let phones: [String]
    public let phoneIDs: [Int]
    public let word2ph: [Int]?
    public let phoneUnits: [GPTSoVITSTextPhoneUnit]
    public let backend: String
}

public struct GPTSoVITSTextSegmentPhoneResult {
    public let segment: GPTSoVITSTextSegment
    public let phoneResult: GPTSoVITSTextPhoneResult
}

public struct GPTSoVITSTextPhonePreprocessResult {
    public let preprocessResult: GPTSoVITSTextPreprocessResult
    public let segmentResults: [GPTSoVITSTextSegmentPhoneResult]
}

public protocol GPTSoVITSTextPhoneBackend {
    func phoneResult(
        for text: String,
        language: GPTSoVITSTextLanguage
    ) throws -> GPTSoVITSTextPhoneResult
}

public enum GPTSoVITSTextPhoneFrontendError: LocalizedError {
    case unsupportedLanguage(String)
    case backendRequired(String)

    public var errorDescription: String? {
        switch self {
        case let .unsupportedLanguage(language):
            return "GPTSoVITSTextPhoneFrontend 暂不支持 language=\(language)。"
        case let .backendRequired(language):
            return "GPTSoVITSTextPhoneFrontend 需要为 language=\(language) 提供后端实现。"
        }
    }
}

public extension GPTSoVITSTextFrontend {
    func preprocessPhoneSegments(
        text: String,
        language: GPTSoVITSTextLanguage,
        splitMethod: GPTSoVITSTextSplitMethod = .cut5,
        backend: (any GPTSoVITSTextPhoneBackend)? = nil,
        g2pwDriver: (any GPTSoVITSG2PWPredicting)? = nil
    ) throws -> GPTSoVITSTextPhonePreprocessResult {
        let preprocessResult = preprocess(text: text, language: language, splitMethod: splitMethod)
        let segmentResults = try preprocessResult.segments.map { segment in
            GPTSoVITSTextSegmentPhoneResult(
                segment: segment,
                phoneResult: try phoneResult(
                    for: segment.text,
                    language: language,
                    backend: backend,
                    g2pwDriver: g2pwDriver
                )
            )
        }
        return GPTSoVITSTextPhonePreprocessResult(
            preprocessResult: preprocessResult,
            segmentResults: segmentResults
        )
    }

    func phoneResult(
        for text: String,
        language: GPTSoVITSTextLanguage,
        backend: (any GPTSoVITSTextPhoneBackend)? = nil,
        g2pwDriver: (any GPTSoVITSG2PWPredicting)? = nil
    ) throws -> GPTSoVITSTextPhoneResult {
        switch language.baseLanguage {
        case "zh":
            if let backend {
                return try backend.phoneResult(for: text, language: language)
            }
            return Self.makeGenericPhoneResult(
                from: try chinesePhoneResult(for: text, language: language, g2pwDriver: g2pwDriver)
            )
        case "yue", "ja", "ko", "en":
            guard let backend else {
                throw GPTSoVITSTextPhoneFrontendError.backendRequired(language.rawValue)
            }
            return try backend.phoneResult(for: text, language: language)
        default:
            throw GPTSoVITSTextPhoneFrontendError.unsupportedLanguage(language.rawValue)
        }
    }

    private static func makeGenericPhoneResult(
        from chineseResult: GPTSoVITSChinesePhoneResult
    ) -> GPTSoVITSTextPhoneResult {
        GPTSoVITSTextPhoneResult(
            sourceText: chineseResult.sourceText,
            normalizedText: chineseResult.normalizedText,
            phones: chineseResult.phones,
            phoneIDs: chineseResult.phoneIDs,
            word2ph: chineseResult.word2ph,
            phoneUnits: chineseResult.phoneUnits.map { unit in
                GPTSoVITSTextPhoneUnit(
                    unitType: makeGenericUnitType(from: unit.unitType),
                    text: unit.text,
                    normText: unit.normText,
                    pos: unit.pos,
                    phones: unit.phones,
                    phoneIDs: unit.phoneIDs,
                    charStart: unit.charStart,
                    charEnd: unit.charEnd,
                    phoneStart: unit.phoneStart,
                    phoneEnd: unit.phoneEnd,
                    phoneCount: unit.phoneCount
                )
            },
            backend: chineseResult.backend
        )
    }

    private static func makeGenericUnitType(
        from chineseUnitType: GPTSoVITSPhoneUnitType
    ) -> GPTSoVITSTextPhoneUnitType {
        switch chineseUnitType {
        case .word:
            return .word
        case .char:
            return .char
        case .punct:
            return .punct
        }
    }
}
