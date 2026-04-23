import CoreML
import Foundation
import NaturalLanguage

public struct EnglishFrontendBundleManifest: Decodable {
    public struct Artifact: Decodable {
        public let target: String
        public let filename: String
        public let path: String
    }

    public struct Artifacts: Decodable {
        public let model: Artifact
        public let runtimeAssets: Artifact

        private enum CodingKeys: String, CodingKey {
            case model
            case runtimeAssets = "runtime_assets"
        }
    }

    public struct Runtime: Decodable {
        public struct Shapes: Decodable {
            public let maxWordLen: Int
            public let maxDecodeLen: Int

            private enum CodingKeys: String, CodingKey {
                case maxWordLen = "max_word_len"
                case maxDecodeLen = "max_decode_len"
            }
        }

        public struct Tokens: Decodable {
            public let decoderStartID: Int
            public let eosID: Int

            private enum CodingKeys: String, CodingKey {
                case decoderStartID = "decoder_start_id"
                case eosID = "eos_id"
            }
        }

        public let shapes: Shapes
        public let tokens: Tokens
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

public struct EnglishFrontendRuntimeAssets: Decodable {
    public struct Homograph: Decodable {
        public let pron1: [String]
        public let pron2: [String]
        public let pos1: String
    }

    public let formatVersion: Int
    public let graphemes: [String]
    public let phonemes: [String]
    public let cmuDict: [String: [String]]
    public let namedDict: [String: [String]]
    public let homographs: [String: Homograph]

    private enum CodingKeys: String, CodingKey {
        case formatVersion = "format_version"
        case graphemes
        case phonemes
        case cmuDict = "cmu_dict"
        case namedDict = "named_dict"
        case homographs
    }
}

public enum EnglishPhoneFrontendError: LocalizedError {
    case invalidBundleType(String)
    case unsupportedLanguage(String)
    case predictorInputTooLong(String)
    case missingModelOutput(String)

    public var errorDescription: String? {
        switch self {
        case let .invalidBundleType(bundleType):
            return "English frontend bundle_type 不匹配: \(bundleType)"
        case let .unsupportedLanguage(language):
            return "EnglishCoreMLPhoneBackend 只支持英文主链，收到 language=\(language)"
        case let .predictorInputTooLong(word):
            return "English OOV predictor 输入过长，当前无法处理单词: \(word)"
        case let .missingModelOutput(name):
            return "English OOV predictor 缺少输出: \(name)"
        }
    }
}

public final class EnglishCoreMLPhoneBackend: GPTSoVITSTextPhoneBackend {
    private struct TokenUnit {
        let text: String
        let range: Range<String.Index>
        let pos: String?
    }

    public let manifest: EnglishFrontendBundleManifest
    public let runtimeAssets: EnglishFrontendRuntimeAssets
    public let model: MLModel

    private let graphemeToID: [String: Int]
    private let phonemeByID: [Int: String]
    private let unkID: Int

    public static func bundleType(at bundleDirectory: URL) -> String? {
        let manifestURL = bundleDirectory.appendingPathComponent("manifest.json")
        guard let data = try? Data(contentsOf: manifestURL) else {
            return nil
        }
        return (try? JSONDecoder().decode(EnglishFrontendBundleManifest.self, from: data))?.bundleType
    }

    public static func isBundleDirectory(_ bundleDirectory: URL) -> Bool {
        bundleType(at: bundleDirectory) == "gpt_sovits_english_frontend_bundle"
    }

    public init(bundleDirectory: URL, configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        let manifestURL = bundleDirectory.appendingPathComponent("manifest.json")
        let manifestData = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(EnglishFrontendBundleManifest.self, from: manifestData)
        guard manifest.bundleType == "gpt_sovits_english_frontend_bundle" else {
            throw EnglishPhoneFrontendError.invalidBundleType(manifest.bundleType)
        }
        let modelURL = bundleDirectory.appendingPathComponent(manifest.artifacts.model.filename)
        let assetsURL = bundleDirectory.appendingPathComponent(manifest.artifacts.runtimeAssets.filename)
        self.manifest = manifest
        self.model = try Self.loadModel(at: modelURL, configuration: configuration)
        self.runtimeAssets = try JSONDecoder().decode(
            EnglishFrontendRuntimeAssets.self,
            from: Data(contentsOf: assetsURL)
        )
        self.graphemeToID = Dictionary(uniqueKeysWithValues: runtimeAssets.graphemes.enumerated().map { ($1, $0) })
        self.phonemeByID = Dictionary(uniqueKeysWithValues: runtimeAssets.phonemes.enumerated().map { ($0, $1) })
        self.unkID = graphemeToID["<unk>"] ?? 1
    }

    public func phoneResult(
        for text: String,
        language: GPTSoVITSTextLanguage
    ) throws -> GPTSoVITSTextPhoneResult {
        guard language.baseLanguage == "en" else {
            throw EnglishPhoneFrontendError.unsupportedLanguage(language.rawValue)
        }
        let normalized = Self.normalizeEnglishText(text)
        let tokens = Self.tokenize(normalized)
        var phoneUnits = [GPTSoVITSTextPhoneUnit]()
        var phones = [String]()
        var cursor = normalized.startIndex
        let unkPhoneID = GPTSoVITSPhoneSymbolAssets.symbolToID["UNK"] ?? 0

        for (tokenIndex, token) in tokens.enumerated() {
            if cursor < token.range.lowerBound {
                let gapText = String(normalized[cursor..<token.range.lowerBound])
                let unitType: GPTSoVITSTextPhoneUnitType = gapText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? .space : .punct
                phoneUnits.append(
                    GPTSoVITSTextPhoneUnit(
                        unitType: unitType,
                        text: gapText,
                        normText: gapText,
                        pos: nil,
                        phones: [],
                        phoneIDs: [],
                        charStart: normalized.distance(from: normalized.startIndex, to: cursor),
                        charEnd: normalized.distance(from: normalized.startIndex, to: token.range.lowerBound),
                        phoneStart: phones.count,
                        phoneEnd: phones.count,
                        phoneCount: 0
                    )
                )
            }

            let tokenPhones = try pronounceToken(token, index: tokenIndex, tokens: tokens)
            let normalizedPhones = Self.normalizePronunciation(tokenPhones)
            let phoneIDs = normalizedPhones.map { GPTSoVITSPhoneSymbolAssets.symbolToID[$0] ?? unkPhoneID }
            let charStart = normalized.distance(from: normalized.startIndex, to: token.range.lowerBound)
            let charEnd = normalized.distance(from: normalized.startIndex, to: token.range.upperBound)
            let phoneStart = phones.count
            phones.append(contentsOf: normalizedPhones)
            let phoneEnd = phones.count
            phoneUnits.append(
                GPTSoVITSTextPhoneUnit(
                    unitType: Self.isWordToken(token.text) ? .word : .punct,
                    text: token.text,
                    normText: Self.isWordToken(token.text) ? token.text.lowercased() : token.text,
                    pos: token.pos,
                    phones: normalizedPhones,
                    phoneIDs: phoneIDs,
                    charStart: charStart,
                    charEnd: charEnd,
                    phoneStart: phoneStart,
                    phoneEnd: phoneEnd,
                    phoneCount: normalizedPhones.count
                )
            )
            cursor = token.range.upperBound
        }

        if cursor < normalized.endIndex {
            let gapText = String(normalized[cursor..<normalized.endIndex])
            let unitType: GPTSoVITSTextPhoneUnitType = gapText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? .space : .punct
            phoneUnits.append(
                GPTSoVITSTextPhoneUnit(
                    unitType: unitType,
                    text: gapText,
                    normText: gapText,
                    pos: nil,
                    phones: [],
                    phoneIDs: [],
                    charStart: normalized.distance(from: normalized.startIndex, to: cursor),
                    charEnd: normalized.count,
                    phoneStart: phones.count,
                    phoneEnd: phones.count,
                    phoneCount: 0
                )
            )
        }

        if phones.count < 4 {
            let prependedPhone = ","
            let prependedPhoneID = GPTSoVITSPhoneSymbolAssets.symbolToID[prependedPhone] ?? unkPhoneID
            phones.insert(prependedPhone, at: 0)
            if !phoneUnits.isEmpty {
                let first = phoneUnits[0]
                phoneUnits[0] = GPTSoVITSTextPhoneUnit(
                    unitType: first.unitType,
                    text: first.text,
                    normText: first.normText,
                    pos: first.pos,
                    phones: [prependedPhone] + first.phones,
                    phoneIDs: [prependedPhoneID] + first.phoneIDs,
                    charStart: first.charStart,
                    charEnd: first.charEnd,
                    phoneStart: 0,
                    phoneEnd: first.phoneEnd + 1,
                    phoneCount: first.phoneCount + 1
                )
                for index in 1..<phoneUnits.count {
                    let unit = phoneUnits[index]
                    phoneUnits[index] = GPTSoVITSTextPhoneUnit(
                        unitType: unit.unitType,
                        text: unit.text,
                        normText: unit.normText,
                        pos: unit.pos,
                        phones: unit.phones,
                        phoneIDs: unit.phoneIDs,
                        charStart: unit.charStart,
                        charEnd: unit.charEnd,
                        phoneStart: unit.phoneStart + 1,
                        phoneEnd: unit.phoneEnd + 1,
                        phoneCount: unit.phoneCount
                    )
                }
            }
        }

        let phoneIDs = phones.map { GPTSoVITSPhoneSymbolAssets.symbolToID[$0] ?? unkPhoneID }
        return GPTSoVITSTextPhoneResult(
            sourceText: text.precomposedStringWithCanonicalMapping,
            normalizedText: normalized.precomposedStringWithCanonicalMapping,
            phones: phones,
            phoneIDs: phoneIDs,
            word2ph: nil,
            phoneUnits: phoneUnits,
            backend: "apple_english_coreml_bundle"
        )
    }

    private func pronounceToken(_ token: TokenUnit, index: Int, tokens: [TokenUnit]) throws -> [String] {
        let lowered = token.text.lowercased()
        if !Self.isWordToken(token.text) {
            return [token.text]
        }
        if token.text.count == 1 {
            if token.text == "A" {
                return ["EY1"]
            }
            return runtimeAssets.cmuDict[lowered] ?? ["UNK"]
        }
        if let homograph = runtimeAssets.homographs[lowered] {
            return resolveHomographPronunciation(
                token: token,
                index: index,
                tokens: tokens,
                homograph: homograph
            )
        }
        return try queryWord(token.text)
    }

    private func resolveHomographPronunciation(
        token: TokenUnit,
        index: Int,
        tokens: [TokenUnit],
        homograph: EnglishFrontendRuntimeAssets.Homograph
    ) -> [String] {
        if Self.matchPOS(token.pos, expectedPrefix: homograph.pos1) {
            if Self.shouldPreferSecondaryHomographPronunciation(
                token: token,
                index: index,
                tokens: tokens,
                expectedPrefix: homograph.pos1
            ) {
                return homograph.pron2
            }
            return homograph.pron1
        }
        if Self.shouldPreferPrimaryHomographPronunciation(
            token: token,
            index: index,
            tokens: tokens,
            expectedPrefix: homograph.pos1
        ) {
            return homograph.pron1
        }
        return homograph.pron2
    }

    private func queryWord(_ token: String) throws -> [String] {
        let lowered = token.lowercased()
        if let pron = runtimeAssets.cmuDict[lowered] {
            return pron
        }
        if token.prefix(1) == token.prefix(1).uppercased(), let pron = runtimeAssets.namedDict[lowered] {
            return pron
        }
        if lowered.count <= 3 {
            var phones = [String]()
            for character in lowered {
                let key = String(character)
                if key == "a" {
                    phones.append("EY1")
                } else if let pron = runtimeAssets.cmuDict[key] {
                    phones.append(contentsOf: pron)
                } else {
                    phones.append("UNK")
                }
            }
            return phones
        }
        if lowered.hasSuffix("'s") {
            let base = String(token.dropLast(2))
            var phones = try queryWord(base)
            if let last = phones.last {
                if ["P", "T", "K", "F", "TH", "HH"].contains(last) {
                    phones.append("S")
                } else if ["S", "Z", "SH", "ZH", "CH", "JH"].contains(last) {
                    phones.append(contentsOf: ["AH0", "Z"])
                } else {
                    phones.append("Z")
                }
            }
            return phones
        }
        return try predictOOV(token: lowered)
    }

    private func predictOOV(token: String) throws -> [String] {
        let ids = try encodeWord(token)
        let inputIDs = try makeInt32Array(shape: [1, manifest.runtime.shapes.maxWordLen], values: ids.paddedIDs)
        let inputLength = try makeInt32Array(shape: [1], values: [ids.length])
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIDs),
            "input_length": MLFeatureValue(multiArray: inputLength),
        ])
        let output = try model.prediction(from: provider)
        guard let phonemeIDs = output.featureValue(for: "phoneme_ids")?.multiArrayValue else {
            throw EnglishPhoneFrontendError.missingModelOutput("phoneme_ids")
        }
        var phones = [String]()
        for index in 0..<phonemeIDs.count {
            let raw = Int(truncating: phonemeIDs[index])
            if raw == manifest.runtime.tokens.eosID {
                break
            }
            if let phoneme = phonemeByID[raw] {
                phones.append(phoneme)
            }
        }
        return phones
    }

    private func encodeWord(_ word: String) throws -> (paddedIDs: [Int32], length: Int32) {
        let tokens = Array(word).map { graphemeToID[String($0)] ?? unkID } + [graphemeToID["</s>"] ?? 2]
        guard tokens.count <= manifest.runtime.shapes.maxWordLen else {
            throw EnglishPhoneFrontendError.predictorInputTooLong(word)
        }
        let padded = tokens + Array(repeating: graphemeToID["<pad>"] ?? 0, count: manifest.runtime.shapes.maxWordLen - tokens.count)
        return (padded.map(Int32.init), Int32(tokens.count))
    }

    private func makeInt32Array(shape: [Int], values: [Int32]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map(NSNumber.init(value:)), dataType: .int32)
        let buffer = array.dataPointer.bindMemory(to: Int32.self, capacity: values.count)
        for (index, value) in values.enumerated() {
            buffer[index] = value
        }
        return array
    }

    private static func loadModel(at url: URL, configuration: MLModelConfiguration) throws -> MLModel {
        try GPTSoVITSCoreMLModelLoader.loadModel(at: url, configuration: configuration)
    }

    private static func tokenize(_ text: String) -> [TokenUnit] {
        let tagger = NLTagger(tagSchemes: [.lexicalClass])
        tagger.string = text
        let pattern = #"[A-Za-z]+(?:'[A-Za-z]+)?|[.,?!\-]"#
        let regex = try? NSRegularExpression(pattern: pattern)
        let range = NSRange(text.startIndex..., in: text)
        let matches = regex?.matches(in: text, options: [], range: range) ?? []
        return matches.compactMap { match in
            guard let tokenRange = Range(match.range, in: text) else { return nil }
            let tokenText = String(text[tokenRange])
            let pos = isWordToken(tokenText)
                ? lexicalPOS(for: tokenRange, in: text, tagger: tagger)
                : nil
            return TokenUnit(text: tokenText, range: tokenRange, pos: pos)
        }
    }

    private static func lexicalPOS(for range: Range<String.Index>, in text: String, tagger: NLTagger) -> String? {
        let (tag, _) = tagger.tag(at: range.lowerBound, unit: .word, scheme: .lexicalClass)
        switch tag {
        case .verb:
            return "V"
        case .noun:
            return "N"
        case .adjective:
            return "J"
        case .adverb:
            return "R"
        default:
            return nil
        }
    }

    private static func matchPOS(_ pos: String?, expectedPrefix: String) -> Bool {
        guard let pos, !expectedPrefix.isEmpty else { return false }
        let normalizedExpected = expectedPrefix.uppercased()
        let normalizedPOS = pos.uppercased()
        if normalizedPOS == normalizedExpected {
            return true
        }
        if normalizedExpected == "ADJ" {
            return normalizedPOS == "J"
        }
        return false
    }

    private static func shouldPreferPrimaryHomographPronunciation(
        token: TokenUnit,
        index: Int,
        tokens: [TokenUnit],
        expectedPrefix: String
    ) -> Bool {
        let normalizedExpected = expectedPrefix.uppercased()
        if normalizedExpected.hasPrefix("N"),
           token.text.first?.isUppercase == true,
           token.text.dropFirst().allSatisfy(\.isLowercase) {
            return true
        }
        return false
    }

    private static func shouldPreferSecondaryHomographPronunciation(
        token: TokenUnit,
        index: Int,
        tokens: [TokenUnit],
        expectedPrefix: String
    ) -> Bool {
        let normalizedExpected = expectedPrefix.uppercased()
        let lowered = token.text.lowercased()
        guard normalizedExpected.hasPrefix("V"),
              !lowered.hasSuffix("ed"),
              !lowered.hasSuffix("ing"),
              !lowered.hasSuffix("en"),
              let previous = previousWordToken(before: index, tokens: tokens),
              let next = nextToken(after: index, tokens: tokens)
        else {
            return false
        }
        if next.text != "." && next.text != "," && next.text != "?" && next.text != "!" {
            return false
        }
        let previousLowered = previous.text.lowercased()
        if isAuxiliaryLike(previousLowered) {
            return false
        }
        if previous.pos == "V" || previousLowered.hasSuffix("ed") || previousLowered.hasSuffix("s") {
            return true
        }
        return false
    }

    private static func previousWordToken(before index: Int, tokens: [TokenUnit]) -> TokenUnit? {
        guard index > 0 else { return nil }
        for probe in stride(from: index - 1, through: 0, by: -1) {
            let token = tokens[probe]
            if isWordToken(token.text) {
                return token
            }
        }
        return nil
    }

    private static func nextToken(after index: Int, tokens: [TokenUnit]) -> TokenUnit? {
        guard index + 1 < tokens.count else { return nil }
        return tokens[index + 1]
    }

    private static func isAuxiliaryLike(_ token: String) -> Bool {
        [
            "am", "is", "are", "was", "were", "be", "been", "being",
            "do", "does", "did",
            "have", "has", "had",
            "can", "could", "may", "might", "must", "shall", "should", "will", "would",
            "to"
        ].contains(token)
    }

    private static func normalizePronunciation(_ pronunciation: [String]) -> [String] {
        let replaceMap = ["'": "-"]
        return pronunciation.compactMap { item in
            if item == "<unk>" {
                return "UNK"
            }
            if [" ", "<pad>", "UW", "</s>", "<s>"].contains(item) {
                return nil
            }
            return replaceMap[item] ?? item
        }
    }

    private static func isWordToken(_ token: String) -> Bool {
        token.range(of: #"[A-Za-z]"#, options: .regularExpression) != nil
    }

    private static func normalizeEnglishText(_ text: String) -> String {
        var value = text
        value = value.replacingOccurrences(of: "[;:：，；]", with: ",", options: .regularExpression)
        value = value.replacingOccurrences(of: #"["’]"#, with: "'", options: .regularExpression)
        value = value.replacingOccurrences(of: "。", with: ".")
        value = value.replacingOccurrences(of: "！", with: "!")
        value = value.replacingOccurrences(of: "？", with: "?")
        value = value.replacingOccurrences(of: "(?i)i\\.e\\.", with: "that is", options: .regularExpression)
        value = value.replacingOccurrences(of: "(?i)e\\.g\\.", with: "for example", options: .regularExpression)
        value = spellOutIntegers(in: value)
        value = splitUppercaseRuns(in: value)
        value = stripAccents(value)
        value = value.replacingOccurrences(of: "%", with: " percent")
        value = value.replacingOccurrences(of: #"[^ A-Za-z'.,?!-]"#, with: "", options: .regularExpression)
        value = replaceConsecutivePunctuation(value)
        return value.precomposedStringWithCanonicalMapping
    }

    private static func stripAccents(_ text: String) -> String {
        text.decomposedStringWithCanonicalMapping.unicodeScalars.filter {
            !$0.properties.isDiacritic
        }.map(String.init).joined()
    }

    private static func spellOutIntegers(in text: String) -> String {
        let regex = try? NSRegularExpression(pattern: #"\d+"#)
        let nsrange = NSRange(text.startIndex..., in: text)
        let matches = (regex?.matches(in: text, options: [], range: nsrange) ?? []).reversed()
        var result = text
        for match in matches {
            guard let range = Range(match.range, in: result) else { continue }
            let token = String(result[range])
            guard let value = Int(token), let spelled = spellOut(value) else { continue }
            result.replaceSubrange(range, with: spelled)
        }
        return result
    }

    private static func spellOut(_ value: Int) -> String? {
        let formatter = NumberFormatter()
        formatter.numberStyle = .spellOut
        formatter.locale = Locale(identifier: "en_US")
        return formatter.string(from: NSNumber(value: value))?.replacingOccurrences(of: "-", with: " ")
    }

    private static func splitUppercaseRuns(in text: String) -> String {
        guard !text.isEmpty else { return text }
        var result = ""
        var previous: Character?
        for character in text {
            if character.isUppercase,
               let previous,
               !previous.isWhitespace {
                result.append(" ")
            }
            result.append(character)
            previous = character
        }
        return result
    }

    private static func replaceConsecutivePunctuation(_ text: String) -> String {
        let regex = try? NSRegularExpression(pattern: #"([,\.?!\-\s])([,\.?!\-])+"#)
        var result = text
        while true {
            let nsrange = NSRange(result.startIndex..., in: result)
            guard let match = regex?.firstMatch(in: result, options: [], range: nsrange),
                  let range = Range(match.range, in: result),
                  let firstRange = Range(match.range(at: 1), in: result) else {
                break
            }
            result.replaceSubrange(range, with: String(result[firstRange]))
        }
        return result
    }
}

private extension Character {
    var isUppercase: Bool {
        unicodeScalars.allSatisfy { CharacterSet.uppercaseLetters.contains($0) }
    }
}
