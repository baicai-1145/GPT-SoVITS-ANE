import Foundation

public enum KoreanPhoneFrontendError: LocalizedError {
    case invalidBundleType(String)
    case unsupportedLanguage(String)
    case unitCountMismatch(source: Int, transformed: Int)
    case unitKindMismatch(source: String, transformed: String)

    public var errorDescription: String? {
        switch self {
        case let .invalidBundleType(bundleType):
            return "韩语 frontend bundle_type 不匹配: \(bundleType)"
        case let .unsupportedLanguage(language):
            return "KoreanPhoneFrontend 只支持韩语主链，收到 language=\(language)"
        case let .unitCountMismatch(source, transformed):
            return "韩语 unit 数不一致: source=\(source), transformed=\(transformed)"
        case let .unitKindMismatch(source, transformed):
            return "韩语 unit 类型不一致: source=\(source), transformed=\(transformed)"
        }
    }
}

public final class KoreanPhoneFrontend: GPTSoVITSTextPhoneBackend {
    private struct SplitUnit {
        let kind: String
        let text: String
    }

    public let bundle: KoreanFrontendBundle

    private let tagger: KoreanMecabTagger
    private let annotator: KoreanG2PAnnotator
    private let transforms: KoreanTextTransforms
    private let unkID: Int

    private let specialSteps: [KoreanFrontendRuntimeAssets.G2pk2EffectiveSpecialAssets.Step]
    private let tableRules: [KoreanFrontendRuntimeAssets.G2pk2Assets.TableRule]

    public static func bundleType(at bundleDirectory: URL) -> String? {
        KoreanFrontendBundle.bundleType(at: bundleDirectory)
    }

    public static func isBundleDirectory(_ bundleDirectory: URL) -> Bool {
        KoreanFrontendBundle.isBundleDirectory(bundleDirectory)
    }

    public init(bundleDirectory: URL) throws {
        let bundle = try KoreanFrontendBundle(bundleDirectory: bundleDirectory)
        self.bundle = bundle
        self.tagger = try KoreanMecabTagger(
            dynamicLibraryURL: bundle.mecabDynamicLibraryURL,
            dictionaryDirectory: bundle.mecabDictionaryDirectory,
            mecabRCFileURL: bundle.mecabRCFileURL
        )
        self.annotator = KoreanG2PAnnotator(tagger: tagger)
        self.transforms = KoreanTextTransforms(bundle: bundle)
        self.unkID = GPTSoVITSPhoneSymbolAssets.symbolToID["UNK"] ?? 0
        self.specialSteps = bundle.runtimeAssets.g2pk2EffectiveSpecialAssets.steps
        self.tableRules = bundle.runtimeAssets.g2pk2Assets.table
    }

    public func phoneResult(
        for text: String,
        language: GPTSoVITSTextLanguage
    ) throws -> GPTSoVITSTextPhoneResult {
        guard language.baseLanguage == "ko" else {
            throw KoreanPhoneFrontendError.unsupportedLanguage(language.rawValue)
        }

        let sourceText = text.precomposedStringWithCanonicalMapping
        let transformed = try transformG2PText(sourceText)
        var sourceUnits = splitUnits(sourceText)
        let transformedUnits = splitPhoneUnits(transformed)

        if sourceUnits.count + 1 == transformedUnits.count,
           let extra = transformedUnits.last,
           extra.kind == "punct",
           sourceUnits.last?.kind != extra.kind || sourceUnits.last?.text != extra.text {
            sourceUnits.append(extra)
        }

        guard sourceUnits.count == transformedUnits.count else {
            throw KoreanPhoneFrontendError.unitCountMismatch(source: sourceUnits.count, transformed: transformedUnits.count)
        }

        var phoneUnits = [GPTSoVITSTextPhoneUnit]()
        var phones = [String]()
        var charCursor = 0
        let sourceCharacters = Array(sourceText)

        for (sourceUnit, transformedUnit) in zip(sourceUnits, transformedUnits) {
            guard sourceUnit.kind == transformedUnit.kind else {
                throw KoreanPhoneFrontendError.unitKindMismatch(source: sourceUnit.kind, transformed: transformedUnit.kind)
            }

            let charStart: Int
            let charEnd: Int
            if sourceUnit.text.isEmpty {
                charStart = charCursor
                charEnd = charCursor
            } else if sourceCharacters.count >= charCursor + sourceUnit.text.count,
                      String(sourceCharacters[charCursor..<(charCursor + sourceUnit.text.count)]) == sourceUnit.text {
                charStart = charCursor
                charCursor += sourceUnit.text.count
                charEnd = charCursor
            } else {
                charStart = min(charCursor, sourceCharacters.count)
                charEnd = charStart
            }

            let unitPhones = transformedUnit.text.unicodeScalars.map { postReplacePhone(String($0)) }
            let phoneIDs = unitPhones.map { GPTSoVITSPhoneSymbolAssets.symbolToID[$0] ?? unkID }
            let phoneStart = phones.count
            phones.append(contentsOf: unitPhones)
            let phoneEnd = phones.count

            phoneUnits.append(
                GPTSoVITSTextPhoneUnit(
                    unitType: unitType(for: sourceUnit.kind),
                    text: sourceUnit.text,
                    normText: transformedUnit.text,
                    pos: nil,
                    phones: unitPhones,
                    phoneIDs: phoneIDs,
                    charStart: charStart,
                    charEnd: charEnd,
                    phoneStart: phoneStart,
                    phoneEnd: phoneEnd,
                    phoneCount: unitPhones.count
                )
            )
        }

        return GPTSoVITSTextPhoneResult(
            sourceText: sourceText,
            normalizedText: sourceText,
            phones: phones,
            phoneIDs: phones.map { GPTSoVITSPhoneSymbolAssets.symbolToID[$0] ?? unkID },
            word2ph: nil,
            phoneUnits: phoneUnits,
            backend: "apple_korean_bundle_native"
        )
    }

    public func debugTransformStages(for text: String) throws -> [String: String] {
        var stages = [String: String]()
        var working = transforms.latinToHangul(text)
        stages["latin"] = working
        working = applyIdioms(to: working)
        stages["idioms"] = working
        working = try annotator.annotate(working)
        stages["annotated"] = working
        working = transforms.numberToHangul(working)
        stages["number"] = working
        working = decomposeToHangulJamo(working)
        stages["decomposed"] = working
        working = applyRegexSteps(specialSteps, to: working)
        stages["special"] = working
        working = removeTags(from: working)
        stages["untagged"] = working
        working = applyTableRules(to: working)
        stages["table"] = working
        working = applyLinkRules(to: working)
        stages["link"] = working
        working = composeHangulJamo(working)
        stages["composed"] = working
        working = transforms.divideHangul(working)
        stages["divided"] = working
        working = transforms.fixG2PK2Error(working)
        stages["fixed"] = working
        return stages
    }

    private func transformG2PText(_ text: String) throws -> String {
        var working = transforms.latinToHangul(text)
        working = applyIdioms(to: working)
        working = try annotator.annotate(working)
        working = transforms.numberToHangul(working)
        working = decomposeToHangulJamo(working)
        working = applyRegexSteps(specialSteps, to: working)
        working = removeTags(from: working)
        working = applyTableRules(to: working)
        working = applyLinkRules(to: working)
        working = composeHangulJamo(working)
        working = transforms.divideHangul(working)
        working = transforms.fixG2PK2Error(working)
        if let last = working.last, Self.compatibilityJamoSet.contains(last) {
            working.append(".")
        }
        return working
    }

    private func applyIdioms(to text: String) -> String {
        var output = text
        for line in bundle.runtimeAssets.g2pk2Assets.idiomsLines {
            let stripped = line.split(separator: "#", maxSplits: 1, omittingEmptySubsequences: false).first.map(String.init) ?? line
            let trimmed = stripped.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmed.contains("===") else { continue }
            let parts = trimmed.components(separatedBy: "===")
            guard parts.count == 2,
                  let regex = try? NSRegularExpression(pattern: parts[0]) else {
                continue
            }
            output = regex.stringByReplacingMatches(
                in: output,
                options: [],
                range: NSRange(output.startIndex..., in: output),
                withTemplate: parts[1]
            )
        }
        return output
    }

    private func decomposeToHangulJamo(_ text: String) -> String {
        var output = String()
        output.reserveCapacity(text.count * 2)
        for character in text {
            guard let scalar = character.unicodeScalars.first,
                  character.unicodeScalars.count == 1,
                  scalar.value >= 0xAC00, scalar.value <= 0xD7A3 else {
                output.append(character)
                continue
            }

            let syllableIndex = Int(scalar.value - 0xAC00)
            let choseongIndex = syllableIndex / 588
            let jungseongIndex = (syllableIndex % 588) / 28
            let jongseongIndex = syllableIndex % 28

            if let choseong = UnicodeScalar(0x1100 + choseongIndex),
               let jungseong = UnicodeScalar(0x1161 + jungseongIndex) {
                output.append(Character(choseong))
                output.append(Character(jungseong))
            }
            if jongseongIndex > 0, let jongseong = UnicodeScalar(0x11A7 + jongseongIndex) {
                output.append(Character(jongseong))
            }
        }
        return output
    }

    private func applyRegexSteps(
        _ steps: [KoreanFrontendRuntimeAssets.G2pk2EffectiveSpecialAssets.Step],
        to text: String
    ) -> String {
        var output = text
        for step in steps {
            for replacement in step.replacements {
                output = applyRegexReplacement(replacement.pattern, replacement: replacement.replacement, to: output)
            }
        }
        return output
    }

    private func removeTags(from text: String) -> String {
        applyRegexReplacement("/[PJEB]", replacement: "", to: text)
    }

    private func applyTableRules(to text: String) -> String {
        var output = text
        for rule in tableRules {
            output = applyRegexReplacement(rule.pattern, replacement: rule.replacement, to: output)
        }
        return output
    }

    private func applyLinkRules(to text: String) -> String {
        var output = text

        for (lhs, rhs) in Self.link1Pairs {
            output = applyRegexReplacement(NSRegularExpression.escapedPattern(for: lhs), replacement: rhs, to: output)
        }
        for (lhs, rhs) in Self.link2Pairs {
            output = applyRegexReplacement(NSRegularExpression.escapedPattern(for: lhs), replacement: rhs, to: output)
        }
        for (lhs, rhs) in Self.link4Pairs {
            output = applyRegexReplacement(NSRegularExpression.escapedPattern(for: lhs), replacement: rhs, to: output)
        }
        return output
    }

    private func composeHangulJamo(_ text: String) -> String {
        let scalars = Array(text.unicodeScalars)
        var normalized = [UnicodeScalar]()
        normalized.reserveCapacity(scalars.count + 8)

        for scalar in scalars {
            if Self.isJungseong(scalar),
               let previous = normalized.last,
               !Self.isChoseong(previous) {
                normalized.append(Self.ieungChoseong)
            }
            normalized.append(scalar)
        }

        var output = String()
        var index = 0
        while index < normalized.count {
            let scalar = normalized[index]
            guard Self.isChoseong(scalar),
                  index + 1 < normalized.count,
                  Self.isJungseong(normalized[index + 1]) else {
                output.unicodeScalars.append(scalar)
                index += 1
                continue
            }

            let choseongIndex = Int(scalar.value - 0x1100)
            let jungseongScalar = normalized[index + 1]
            let jungseongIndex = Int(jungseongScalar.value - 0x1161)
            var jongseongIndex = 0
            var consumed = 2

            if index + 2 < normalized.count,
               Self.isJongseong(normalized[index + 2]),
               (index + 3 >= normalized.count || !Self.isJungseong(normalized[index + 3])) {
                jongseongIndex = Int(normalized[index + 2].value - 0x11A7)
                consumed = 3
            }

            if let syllable = UnicodeScalar(0xAC00 + choseongIndex * 588 + jungseongIndex * 28 + jongseongIndex) {
                output.unicodeScalars.append(syllable)
                index += consumed
            } else {
                output.unicodeScalars.append(scalar)
                index += 1
            }
        }
        return output
    }

    private func applyRegexReplacement(
        _ pattern: String,
        replacement: String,
        to text: String
    ) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return text
        }
        let template = Self.regexTemplate(fromPythonReplacement: replacement)
        return regex.stringByReplacingMatches(
            in: text,
            options: [],
            range: NSRange(text.startIndex..., in: text),
            withTemplate: template
        )
    }

    private func splitUnits(_ text: String) -> [SplitUnit] {
        guard !text.isEmpty else { return [] }
        let characters = Array(text)
        var units = [SplitUnit]()
        var cursor = 0
        while cursor < characters.count {
            let kind = Self.separatorKind(for: characters[cursor])
            var end = cursor + 1
            if kind != "word" {
                while end < characters.count, Self.separatorKind(for: characters[end]) == kind {
                    end += 1
                }
            } else {
                while end < characters.count, Self.separatorKind(for: characters[end]) == "word" {
                    end += 1
                }
            }
            units.append(SplitUnit(kind: kind, text: String(characters[cursor..<end])))
            cursor = end
        }
        return units
    }

    private func splitPhoneUnits(_ text: String) -> [SplitUnit] {
        guard !text.isEmpty else { return [] }
        let scalars = Array(text.unicodeScalars)
        var units = [SplitUnit]()
        var cursor = 0
        while cursor < scalars.count {
            let current = Character(scalars[cursor])
            let kind = Self.separatorKind(for: current)
            var end = cursor + 1
            if kind != "word" {
                while end < scalars.count, Self.separatorKind(for: Character(scalars[end])) == kind {
                    end += 1
                }
            } else {
                while end < scalars.count, Self.separatorKind(for: Character(scalars[end])) == "word" {
                    end += 1
                }
            }
            let text = String(String.UnicodeScalarView(scalars[cursor..<end]))
            units.append(SplitUnit(kind: kind, text: text))
            cursor = end
        }
        return units
    }

    private func postReplacePhone(_ phone: String) -> String {
        if let mapped = bundle.runtimeAssets.projectAssets.postReplaceMap[phone] {
            return mapped
        }
        if GPTSoVITSPhoneSymbolAssets.symbolToID[phone] != nil {
            return phone
        }
        return "停"
    }

    private func unitType(for kind: String) -> GPTSoVITSTextPhoneUnitType {
        switch kind {
        case "space":
            return .space
        case "punct":
            return .punct
        default:
            return .word
        }
    }

    private static func regexTemplate(fromPythonReplacement replacement: String) -> String {
        var output = replacement
        for index in stride(from: 9, through: 1, by: -1) {
            output = output.replacingOccurrences(of: "\\\(index)", with: "$\(index)")
        }
        return output
    }

    private static func separatorKind(for character: Character) -> String {
        if character.isWhitespace {
            return "space"
        }
        if punctuationCharacters.contains(character) {
            return "punct"
        }
        return "word"
    }

    private static let punctuationCharacters: Set<Character> = ["：", "；", "，", "。", "！", "？", "\n", "·", "、", ".", ",", "!", "?", ";", ":"]

    private static let compatibilityJamoSet: Set<Character> = Set("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
    private static let ieungChoseong = UnicodeScalar(0x110B)!

    private static let link1Pairs = [
        ("ᆨᄋ", "ᄀ"), ("ᆩᄋ", "ᄁ"), ("ᆫᄋ", "ᄂ"), ("ᆮᄋ", "ᄃ"), ("ᆯᄋ", "ᄅ"),
        ("ᆷᄋ", "ᄆ"), ("ᆸᄋ", "ᄇ"), ("ᆺᄋ", "ᄉ"), ("ᆻᄋ", "ᄊ"), ("ᆽᄋ", "ᄌ"),
        ("ᆾᄋ", "ᄎ"), ("ᆿᄋ", "ᄏ"), ("ᇀᄋ", "ᄐ"), ("ᇁᄋ", "ᄑ"),
    ]

    private static let link2Pairs = [
        ("ᆪᄋ", "ᆨᄊ"), ("ᆬᄋ", "ᆫᄌ"), ("ᆰᄋ", "ᆯᄀ"), ("ᆱᄋ", "ᆯᄆ"), ("ᆲᄋ", "ᆯᄇ"),
        ("ᆳᄋ", "ᆯᄊ"), ("ᆴᄋ", "ᆯᄐ"), ("ᆵᄋ", "ᆯᄑ"), ("ᆹᄋ", "ᆸᄊ"),
    ]

    private static let link4Pairs = [
        ("ᇂᄋ", "ᄋ"), ("ᆭᄋ", "ᄂ"), ("ᆶᄋ", "ᄅ"),
    ]

    private static func isChoseong(_ scalar: UnicodeScalar) -> Bool {
        scalar.value >= 0x1100 && scalar.value <= 0x1112
    }

    private static func isJungseong(_ scalar: UnicodeScalar) -> Bool {
        scalar.value >= 0x1161 && scalar.value <= 0x1175
    }

    private static func isJongseong(_ scalar: UnicodeScalar) -> Bool {
        scalar.value >= 0x11A8 && scalar.value <= 0x11C2
    }
}
