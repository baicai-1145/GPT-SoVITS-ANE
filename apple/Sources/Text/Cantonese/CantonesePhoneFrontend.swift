import Foundation

public struct CantoneseFrontendBundleManifest: Decodable {
    public struct Artifact: Decodable {
        public let target: String
        public let filename: String
        public let path: String
    }

    public struct Artifacts: Decodable {
        public let runtimeAssets: Artifact

        private enum CodingKeys: String, CodingKey {
            case runtimeAssets = "runtime_assets"
        }
    }

    public let bundleType: String
    public let bundleDir: String
    public let artifacts: Artifacts

    private enum CodingKeys: String, CodingKey {
        case bundleType = "bundle_type"
        case bundleDir = "bundle_dir"
        case artifacts
    }
}

public struct CantoneseFrontendRuntimeAssets: Decodable {
    public let formatVersion: Int
    public let maxKeyLength: Int
    public let phraseLookup: [String: [String]]
    public let initials: [String]
    public let punctuation: [String]
    public let replacementMap: [String: String]
    public let traditionalToSimplifiedMap: [String: String]
    public let digitMap: [String: String]
    public let operatorMap: [String: String]

    private enum CodingKeys: String, CodingKey {
        case formatVersion = "format_version"
        case maxKeyLength = "max_key_length"
        case phraseLookup = "phrase_lookup"
        case initials
        case punctuation
        case replacementMap = "replacement_map"
        case traditionalToSimplifiedMap = "traditional_to_simplified_map"
        case digitMap = "digit_map"
        case operatorMap = "operator_map"
    }
}

public enum CantonesePhoneFrontendError: LocalizedError {
    case invalidBundleType(String)
    case unsupportedLanguage(String)
    case missingPronunciation(String)
    case syllableCountMismatch(text: String, syllables: [String])
    case invalidSyllable(String)

    public var errorDescription: String? {
        switch self {
        case let .invalidBundleType(bundleType):
            return "粤语 frontend bundle_type 不匹配: \(bundleType)"
        case let .unsupportedLanguage(language):
            return "CantonesePhoneFrontend 只支持粤语主链，收到 language=\(language)"
        case let .missingPronunciation(text):
            return "粤语文本前端缺少读音: \(text)"
        case let .syllableCountMismatch(text, syllables):
            return "粤语词条 syllable 数与文本长度不一致: text=\(text), syllables=\(syllables)"
        case let .invalidSyllable(syllable):
            return "无法把 jyutping syllable 映射到 GPT-SoVITS phones: \(syllable)"
        }
    }
}

public final class CantonesePhoneFrontend: GPTSoVITSTextPhoneBackend {
    private struct PhraseMatch {
        let text: String
        let syllables: [String]
    }

    public let manifest: CantoneseFrontendBundleManifest
    public let runtimeAssets: CantoneseFrontendRuntimeAssets

    private let punctuationSet: Set<Character>
    private let replacementKeys: [String]
    private let operatorKeys: [String]

    public static func bundleType(at bundleDirectory: URL) -> String? {
        let manifestURL = bundleDirectory.appendingPathComponent("manifest.json")
        guard let data = try? Data(contentsOf: manifestURL) else {
            return nil
        }
        return (try? JSONDecoder().decode(CantoneseFrontendBundleManifest.self, from: data))?.bundleType
    }

    public static func isBundleDirectory(_ bundleDirectory: URL) -> Bool {
        bundleType(at: bundleDirectory) == "gpt_sovits_yue_frontend_bundle"
    }

    public init(bundleDirectory: URL) throws {
        let manifestURL = bundleDirectory.appendingPathComponent("manifest.json")
        let manifestData = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(CantoneseFrontendBundleManifest.self, from: manifestData)
        guard manifest.bundleType == "gpt_sovits_yue_frontend_bundle" else {
            throw CantonesePhoneFrontendError.invalidBundleType(manifest.bundleType)
        }
        let assetsURL = bundleDirectory.appendingPathComponent(manifest.artifacts.runtimeAssets.filename)
        self.manifest = manifest
        self.runtimeAssets = try JSONDecoder().decode(
            CantoneseFrontendRuntimeAssets.self,
            from: Data(contentsOf: assetsURL)
        )
        self.punctuationSet = Set(runtimeAssets.punctuation.compactMap(\.first))
        self.replacementKeys = runtimeAssets.replacementMap.keys.sorted { lhs, rhs in
            if lhs.count == rhs.count {
                return lhs < rhs
            }
            return lhs.count > rhs.count
        }
        self.operatorKeys = runtimeAssets.operatorMap.keys.sorted { lhs, rhs in
            if lhs.count == rhs.count {
                return lhs < rhs
            }
            return lhs.count > rhs.count
        }
    }

    public func phoneResult(
        for text: String,
        language: GPTSoVITSTextLanguage
    ) throws -> GPTSoVITSTextPhoneResult {
        guard language.baseLanguage == "yue" else {
            throw CantonesePhoneFrontendError.unsupportedLanguage(language.rawValue)
        }

        let sourceText = text.precomposedStringWithCanonicalMapping
        let normalizedText = normalizeText(sourceText)
        let unkID = GPTSoVITSPhoneSymbolAssets.symbolToID["UNK"] ?? 0
        guard !normalizedText.isEmpty else {
            return GPTSoVITSTextPhoneResult(
                sourceText: sourceText,
                normalizedText: normalizedText,
                phones: [],
                phoneIDs: [],
                word2ph: [],
                phoneUnits: [],
                backend: "apple_yue_bundle_lookup"
            )
        }

        let characters = Array(normalizedText)
        var phones = [String]()
        var word2ph = [Int]()
        var phoneUnits = [GPTSoVITSTextPhoneUnit]()
        var index = 0

        while index < characters.count {
            let current = characters[index]
            if punctuationSet.contains(current) {
                let phone = String(current)
                let phoneID = GPTSoVITSPhoneSymbolAssets.symbolToID[phone] ?? unkID
                word2ph.append(1)
                phoneUnits.append(
                    GPTSoVITSTextPhoneUnit(
                        unitType: .punct,
                        text: phone,
                        normText: phone,
                        pos: nil,
                        phones: [phone],
                        phoneIDs: [phoneID],
                        charStart: index,
                        charEnd: index + 1,
                        phoneStart: phones.count,
                        phoneEnd: phones.count + 1,
                        phoneCount: 1
                    )
                )
                phones.append(phone)
                index += 1
                continue
            }

            guard let phrase = longestMatch(in: characters, start: index) else {
                throw CantonesePhoneFrontendError.missingPronunciation(String(current))
            }

            if phrase.text.count > 1 {
                guard phrase.text.count == phrase.syllables.count else {
                    throw CantonesePhoneFrontendError.syllableCountMismatch(
                        text: phrase.text,
                        syllables: phrase.syllables
                    )
                }
                for offset in 0..<phrase.syllables.count {
                    let charText = String(characters[index + offset])
                    let unitResult = try buildUnit(
                        text: charText,
                        normText: charText,
                        syllables: [phrase.syllables[offset]],
                        charStart: index + offset,
                        charEnd: index + offset + 1,
                        unkID: unkID,
                        phoneStart: phones.count
                    )
                    phones.append(contentsOf: unitResult.unit.phones)
                    word2ph.append(contentsOf: unitResult.word2ph)
                    phoneUnits.append(unitResult.unit)
                }
            } else {
                let charText = phrase.text
                let unitResult = try buildUnit(
                    text: charText,
                    normText: charText,
                    syllables: phrase.syllables,
                    charStart: index,
                    charEnd: index + 1,
                    unkID: unkID,
                    phoneStart: phones.count
                )
                phones.append(contentsOf: unitResult.unit.phones)
                word2ph.append(contentsOf: unitResult.word2ph)
                phoneUnits.append(unitResult.unit)
            }

            index += phrase.text.count
        }

        let phoneIDs = phones.map { GPTSoVITSPhoneSymbolAssets.symbolToID[$0] ?? unkID }
        return GPTSoVITSTextPhoneResult(
            sourceText: sourceText,
            normalizedText: normalizedText,
            phones: phones,
            phoneIDs: phoneIDs,
            word2ph: word2ph,
            phoneUnits: phoneUnits,
            backend: "apple_yue_bundle_lookup"
        )
    }

    private func normalizeText(_ text: String) -> String {
        var working = text.precomposedStringWithCanonicalMapping
            .replacingOccurrences(of: "嗯", with: "恩")
            .replacingOccurrences(of: "呣", with: "母")

        for key in operatorKeys {
            guard let value = runtimeAssets.operatorMap[key] else { continue }
            working = working.replacingOccurrences(of: key, with: value)
        }
        for key in replacementKeys {
            guard let value = runtimeAssets.replacementMap[key] else { continue }
            working = working.replacingOccurrences(of: key, with: value)
        }

        var normalized = String()
        normalized.reserveCapacity(working.count)
        for character in working {
            let raw = String(character)
            if let mapped = runtimeAssets.traditionalToSimplifiedMap[raw] {
                normalized += mapped
            } else if let mapped = runtimeAssets.digitMap[raw] {
                normalized += mapped
            } else {
                normalized += raw
            }
        }

        return String(
            normalized.filter { character in
                Self.isChineseCharacter(character) || punctuationSet.contains(character)
            }
        )
    }

    private func longestMatch(in characters: [Character], start: Int) -> PhraseMatch? {
        let upperBound = min(characters.count - start, runtimeAssets.maxKeyLength)
        guard upperBound > 0 else { return nil }
        for length in stride(from: upperBound, through: 1, by: -1) {
            let candidate = String(characters[start..<(start + length)])
            if let syllables = runtimeAssets.phraseLookup[candidate] {
                return PhraseMatch(text: candidate, syllables: syllables)
            }
        }
        return nil
    }

    private func buildUnit(
        text: String,
        normText: String,
        syllables: [String],
        charStart: Int,
        charEnd: Int,
        unkID: Int,
        phoneStart: Int
    ) throws -> (unit: GPTSoVITSTextPhoneUnit, word2ph: [Int]) {
        var phones = [String]()
        var word2ph = [Int]()
        for syllable in syllables {
            let syllablePhones = try Self.jyutpingSyllableToPhones(
                syllable,
                initials: runtimeAssets.initials,
                punctuationSet: punctuationSet
            )
            phones.append(contentsOf: syllablePhones)
            word2ph.append(syllablePhones.count)
        }
        let phoneIDs = phones.map { GPTSoVITSPhoneSymbolAssets.symbolToID[$0] ?? unkID }
        let unit = GPTSoVITSTextPhoneUnit(
            unitType: .word,
            text: text,
            normText: normText,
            pos: nil,
            phones: phones,
            phoneIDs: phoneIDs,
            charStart: charStart,
            charEnd: charEnd,
            phoneStart: phoneStart,
            phoneEnd: phoneStart + phones.count,
            phoneCount: phones.count
        )
        return (unit, word2ph)
    }

    private static func jyutpingSyllableToPhones(
        _ syllable: String,
        initials: [String],
        punctuationSet: Set<Character>
    ) throws -> [String] {
        if syllable.count == 1, let only = syllable.first, punctuationSet.contains(only) {
            return [syllable]
        }
        if syllable == "_" {
            return [syllable]
        }

        let tone: Int
        let base: String
        if let last = syllable.last, let toneValue = Int(String(last)) {
            tone = toneValue
            base = String(syllable.dropLast())
        } else {
            tone = 0
            base = syllable
        }

        for initial in initials {
            guard base.hasPrefix(initial) else { continue }
            let initialsFinals: [String]
            let tones: [Int]
            if base.hasPrefix("nga") {
                let head = String(base.prefix(2))
                let tail = String(base.dropFirst(2))
                initialsFinals = [head, tail.isEmpty ? String(base.suffix(1)) : tail]
                tones = [-1, tone]
            } else {
                let tail = String(base.dropFirst(initial.count))
                let final = tail.isEmpty ? String(initial.suffix(1)) : tail
                initialsFinals = [initial, final]
                tones = [-1, tone]
            }

            var phones = [String]()
            for (part, toneValue) in zip(initialsFinals, tones) {
                var token = toneValue == -1 || toneValue == 0 ? part : "\(part)\(toneValue)"
                if !Self.isPunctuationToken(token, punctuationSet: punctuationSet) {
                    token = "Y\(token)"
                }
                phones.append(token)
            }
            return phones
        }

        throw CantonesePhoneFrontendError.invalidSyllable(syllable)
    }

    private static func isPunctuationToken(_ token: String, punctuationSet: Set<Character>) -> Bool {
        token.count == 1 && token.first.map { punctuationSet.contains($0) } == true
    }

    private static func isChineseCharacter(_ character: Character) -> Bool {
        character.unicodeScalars.allSatisfy { scalar in
            (0x4E00...0x9FA5).contains(Int(scalar.value))
        }
    }
}
