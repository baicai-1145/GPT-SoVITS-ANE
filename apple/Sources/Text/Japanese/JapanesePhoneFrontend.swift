import Foundation

public enum JapanesePhoneFrontendError: LocalizedError {
    case unsupportedLanguage(String)
    case candidateAnalysisFailed(String)

    public var errorDescription: String? {
        switch self {
        case let .unsupportedLanguage(language):
            return "JapanesePhoneFrontend 只支持日语主链，收到 language=\(language)"
        case let .candidateAnalysisFailed(text):
            return "日语 frontend 无法为候选词生成读音: \(text)"
        }
    }
}

public final class JapanesePhoneFrontend: GPTSoVITSTextPhoneBackend {
    private struct RawUnit {
        let unitType: GPTSoVITSTextPhoneUnitType
        let text: String
        let normText: String
        let pos: String?
        let phones: [String]
        let charStart: Int
        let charEnd: Int
    }

    private let bundle: JapaneseFrontendBundle
    private let openJTalk: JapaneseOpenJTalk
    private let japaneseCharacterRegex: NSRegularExpression
    private let punctuationCharacters: Set<Character> = ["!", "?", "…", ",", ".", "-"]
    private let prosodyMarks: Set<String>
    private let unkID: Int
    private var g2pCache = [String: [String]]()

    public init(bundleDirectory: URL) throws {
        let bundle = try JapaneseFrontendBundle(bundleDirectory: bundleDirectory)
        self.bundle = bundle
        self.openJTalk = try JapaneseOpenJTalk(
            dynamicLibraryURL: bundle.openjtalkDynamicLibraryURL,
            dictionaryDirectory: bundle.openjtalkDictionaryDirectory,
            userDictionaryDirectory: bundle.jaUserDictionaryDirectory
        )
        self.japaneseCharacterRegex = try NSRegularExpression(
            pattern: bundle.runtimeAssets.projectAssets.japaneseCharactersPattern
        )
        self.prosodyMarks = Set(bundle.runtimeAssets.projectAssets.prosodyMarks)
        self.unkID = GPTSoVITSPhoneSymbolAssets.symbolToID["UNK"] ?? 0
    }

    public func phoneResult(
        for text: String,
        language: GPTSoVITSTextLanguage
    ) throws -> GPTSoVITSTextPhoneResult {
        guard language.baseLanguage == "ja" else {
            throw JapanesePhoneFrontendError.unsupportedLanguage(language.rawValue)
        }

        let sourceText = text.precomposedStringWithCanonicalMapping
        let normalizedText = replaceConsecutivePunctuation(sourceText)
        let transformedText = transformSourceText(normalizedText)
        let rawUnits = try g2pWithPhoneUnits(
            originalText: normalizedText,
            transformedText: transformedText,
            withProsody: true
        )

        var phones = [String]()
        var phoneUnits = [GPTSoVITSTextPhoneUnit]()
        for rawUnit in rawUnits {
            let mappedPhones = rawUnit.phones.map(normalizePhone)
            let phoneStart = phones.count
            phones.append(contentsOf: mappedPhones)
            let phoneEnd = phones.count
            phoneUnits.append(
                GPTSoVITSTextPhoneUnit(
                    unitType: rawUnit.unitType,
                    text: rawUnit.text,
                    normText: rawUnit.normText,
                    pos: rawUnit.pos,
                    phones: mappedPhones,
                    phoneIDs: mappedPhones.map { GPTSoVITSPhoneSymbolAssets.symbolToID[$0] ?? unkID },
                    charStart: rawUnit.charStart,
                    charEnd: rawUnit.charEnd,
                    phoneStart: phoneStart,
                    phoneEnd: phoneEnd,
                    phoneCount: mappedPhones.count
                )
            )
        }

        return GPTSoVITSTextPhoneResult(
            sourceText: sourceText,
            normalizedText: normalizedText,
            phones: phones,
            phoneIDs: phones.map { GPTSoVITSPhoneSymbolAssets.symbolToID[$0] ?? unkID },
            word2ph: nil,
            phoneUnits: phoneUnits,
            backend: "apple_japanese_bundle_native"
        )
    }

    private func g2pWithPhoneUnits(
        originalText: String,
        transformedText: String,
        withProsody: Bool
    ) throws -> [RawUnit] {
        let segments = splitSentenceAndMarks(transformedText)
        let originalCharacters = Array(originalText)
        var units = [RawUnit]()
        var transformedCursor = 0
        var originalCursor = 0

        for segment in segments {
            switch segment {
            case let .sentence(sentence):
                guard matchesJapaneseCharacters(sentence) else {
                    transformedCursor += sentence.count
                    continue
                }
                let analysis = try openJTalk.analyze(sentence)
                let rawPhones = phonesFromLabels(analysis.labels, withProsody: withProsody)
                let fullTokens = rawPhones.map(postReplacePhone)
                let sentenceUnits = try alignSentenceUnits(
                    sentence: sentence,
                    features: analysis.features,
                    fullTokens: fullTokens
                )
                units.append(contentsOf: assignSentenceUnits(
                    sentenceUnits,
                    transformedSentence: sentence,
                    transformedCursor: transformedCursor,
                    originalCharacters: originalCharacters,
                    originalCursor: &originalCursor
                ))
                transformedCursor += sentence.count

            case let .mark(mark):
                let markLength = mark.count
                if mark == " " {
                    transformedCursor += markLength
                    originalCursor += markLength
                    continue
                }
                let cleanedMark = mark.replacingOccurrences(of: " ", with: "")
                let charStart = min(originalCursor, originalCharacters.count)
                originalCursor = min(originalCursor + markLength, originalCharacters.count)
                let charEnd = originalCursor
                units.append(
                    RawUnit(
                        unitType: .punct,
                        text: mark,
                        normText: cleanedMark,
                        pos: nil,
                        phones: [postReplacePhone(cleanedMark)],
                        charStart: charStart,
                        charEnd: charEnd
                    )
                )
                transformedCursor += markLength
            }
        }

        return units
    }

    private func alignSentenceUnits(
        sentence: String,
        features: [JapaneseOpenJTalk.Feature],
        fullTokens: [String]
    ) throws -> [SentenceUnit] {
        if let units = try alignFrontendWords(features, fullTokens: fullTokens, wordIndex: 0, cursor: 0) {
            return units
        }
        return [
            SentenceUnit(
                unitType: .wordGroup,
                text: sentence,
                normText: sentence,
                pos: nil,
                phones: fullTokens,
                transformedCharStart: 0,
                transformedCharEnd: sentence.count
            )
        ]
    }

    private struct SentenceUnit {
        let unitType: GPTSoVITSTextPhoneUnitType
        let text: String
        let normText: String
        let pos: String?
        let phones: [String]
        let transformedCharStart: Int
        let transformedCharEnd: Int
    }

    private func alignFrontendWords(
        _ features: [JapaneseOpenJTalk.Feature],
        fullTokens: [String],
        wordIndex: Int,
        cursor: Int
    ) throws -> [SentenceUnit]? {
        if wordIndex >= features.count {
            var trailingEnd = cursor
            while trailingEnd < fullTokens.count && isProsodyMark(fullTokens[trailingEnd]) {
                trailingEnd += 1
            }
            if trailingEnd != fullTokens.count {
                return nil
            }
            return buildProsodyUnits(fullTokens: fullTokens, start: cursor, end: trailingEnd)
        }

        let feature = features[wordIndex]
        for candidatePhones in try frontendWordPhoneCandidates(feature) {
            guard let aligned = alignWordCandidate(fullTokens, cursor: cursor, candidatePhones: candidatePhones) else {
                continue
            }
            let nextCursor = aligned.nextCursor
            var boundaryEnd = nextCursor
            while boundaryEnd < fullTokens.count && isProsodyMark(fullTokens[boundaryEnd]) {
                boundaryEnd += 1
            }
            guard let rest = try alignFrontendWords(features, fullTokens: fullTokens, wordIndex: wordIndex + 1, cursor: boundaryEnd) else {
                continue
            }
            return [
                SentenceUnit(
                    unitType: .word,
                    text: feature.string,
                    normText: preferredNormText(for: feature),
                    pos: feature.pos,
                    phones: aligned.unitTokens,
                    transformedCharStart: 0,
                    transformedCharEnd: 0
                ),
            ] + buildProsodyUnits(fullTokens: fullTokens, start: nextCursor, end: boundaryEnd) + rest
        }

        if wordIndex == features.count - 1, cursor < fullTokens.count {
            return [
                SentenceUnit(
                    unitType: .word,
                    text: feature.string,
                    normText: preferredNormText(for: feature),
                    pos: feature.pos,
                    phones: Array(fullTokens[cursor...]),
                    transformedCharStart: 0,
                    transformedCharEnd: 0
                ),
            ]
        }
        return nil
    }

    private func assignSentenceUnits(
        _ units: [SentenceUnit],
        transformedSentence: String,
        transformedCursor: Int,
        originalCharacters: [Character],
        originalCursor: inout Int
    ) -> [RawUnit] {
        var assigned = [RawUnit]()
        var transformedUnitCursor = transformedCursor

        for unit in units {
            let charStart: Int
            let charEnd: Int
            if unit.unitType == .prosody {
                charStart = originalCursor
                charEnd = originalCursor
            } else {
                let length = unit.text.count
                let consumedStart = min(originalCursor, originalCharacters.count)
                originalCursor = min(originalCursor + length, originalCharacters.count)
                let consumedEnd = originalCursor
                if unit.unitType == .wordGroup {
                    charStart = 0
                    charEnd = 0
                } else {
                    charStart = consumedStart
                    charEnd = consumedEnd
                }
                transformedUnitCursor += length
            }

            assigned.append(
                RawUnit(
                    unitType: unit.unitType,
                    text: unit.text,
                    normText: unit.normText,
                    pos: unit.pos,
                    phones: unit.phones,
                    charStart: charStart,
                    charEnd: charEnd
                )
            )
        }

        if transformedUnitCursor < transformedCursor + transformedSentence.count {
            originalCursor = min(originalCursor + (transformedCursor + transformedSentence.count - transformedUnitCursor), originalCharacters.count)
        }
        return assigned
    }

    private func buildProsodyUnits(fullTokens: [String], start: Int, end: Int) -> [SentenceUnit] {
        guard start < end else { return [] }
        return (start..<end).map { index in
            SentenceUnit(
                unitType: .prosody,
                text: fullTokens[index],
                normText: fullTokens[index],
                pos: nil,
                phones: [fullTokens[index]],
                transformedCharStart: 0,
                transformedCharEnd: 0
            )
        }
    }

    private func alignWordCandidate(
        _ fullTokens: [String],
        cursor: Int,
        candidatePhones: [String]
    ) -> (unitTokens: [String], nextCursor: Int)? {
        var probe = cursor
        var matched = 0
        var unitTokens = [String]()

        while probe < fullTokens.count && matched < candidatePhones.count {
            let current = fullTokens[probe]
            if phonesEquivalent(current, candidatePhones[matched]) {
                unitTokens.append(current)
                matched += 1
                probe += 1
                continue
            }
            if isProsodyMark(current) {
                unitTokens.append(current)
                probe += 1
                continue
            }
            return nil
        }

        if matched != candidatePhones.count {
            return nil
        }
        return (unitTokens, probe)
    }

    private func frontendWordPhoneCandidates(_ feature: JapaneseOpenJTalk.Feature) throws -> [[String]] {
        var rawCandidates = [String]()
        for value in [feature.string, feature.pron, feature.read] {
            let cleaned = value.replacingOccurrences(of: "’", with: "").replacingOccurrences(of: "'", with: "")
            if cleaned.isEmpty || cleaned == "*" || rawCandidates.contains(cleaned) {
                continue
            }
            rawCandidates.append(cleaned)
        }

        var candidates = [[String]]()
        for value in rawCandidates {
            let phones = try g2pTokens(for: value, withProsody: false).map(postReplacePhone)
            if !phones.isEmpty && !candidates.contains(phones) {
                candidates.append(phones)
            }
        }
        return candidates
    }

    private func g2pTokens(for text: String, withProsody: Bool) throws -> [String] {
        if !withProsody, let cached = g2pCache[text] {
            return cached
        }
        let analysis = try openJTalk.analyze(text)
        let tokens = phonesFromLabels(analysis.labels, withProsody: withProsody)
        if !withProsody {
            g2pCache[text] = tokens
        }
        return tokens
    }

    private func phonesFromLabels(_ labels: [String], withProsody: Bool) -> [String] {
        if withProsody {
            var phones = [String]()
            phones.reserveCapacity(labels.count)
            let count = labels.count
            for index in 0..<count {
                let currentLabel = labels[index]
                var p3 = capture(#"\-(.*?)\+"#, in: currentLabel) ?? ""
                if ["A", "E", "I", "O", "U"].contains(p3) {
                    p3 = p3.lowercased()
                }
                if p3 == "sil" {
                    if index == 0 {
                        phones.append("^")
                    } else if index == count - 1 {
                        let e3 = numericFeature(#"!(\d+)_"#, in: currentLabel)
                        if e3 == 0 {
                            phones.append("$")
                        } else if e3 == 1 {
                            phones.append("?")
                        }
                    }
                    continue
                } else if p3 == "pau" {
                    phones.append("_")
                    continue
                } else {
                    phones.append(p3)
                }

                let a1 = numericFeature(#"/A:([0-9\-]+)\+"#, in: currentLabel)
                let a2 = numericFeature(#"\+(\d+)\+"#, in: currentLabel)
                let a3 = numericFeature(#"\+(\d+)/"#, in: currentLabel)
                let f1 = numericFeature(#"/F:(\d+)_"#, in: currentLabel)
                let nextA2 = index + 1 < count ? numericFeature(#"\+(\d+)\+"#, in: labels[index + 1]) : -50

                if a3 == 1 && nextA2 == 1 && "aeiouAEIOUNcl".contains(p3) {
                    phones.append("#")
                } else if a1 == 0 && nextA2 == a2 + 1 && a2 != f1 {
                    phones.append("]")
                } else if a2 == 1 && nextA2 == 2 {
                    phones.append("[")
                }
            }
            if phones.count >= 2 {
                return Array(phones.dropFirst().dropLast())
            }
            return []
        }

        var phones = [String]()
        for label in labels {
            var p3 = capture(#"\-(.*?)\+"#, in: label) ?? ""
            if ["A", "E", "I", "O", "U"].contains(p3) {
                p3 = p3.lowercased()
            }
            if p3 == "sil" || p3 == "pau" || p3.isEmpty {
                continue
            }
            phones.append(p3)
        }
        return phones
    }

    private func splitSentenceAndMarks(_ text: String) -> [TextSegment] {
        var segments = [TextSegment]()
        var sentence = String()
        for character in text {
            if matchesJapaneseCharacters(String(character)) {
                sentence.append(character)
                continue
            }
            if !sentence.isEmpty {
                segments.append(.sentence(sentence))
                sentence.removeAll(keepingCapacity: true)
            }
            segments.append(.mark(String(character)))
        }
        if !sentence.isEmpty {
            segments.append(.sentence(sentence))
        }
        return segments
    }

    private enum TextSegment {
        case sentence(String)
        case mark(String)
    }

    private func transformSourceText(_ text: String) -> String {
        var output = text
        for replacement in bundle.runtimeAssets.projectAssets.symbolsToJapanese {
            output = applyRegexReplacement(replacement.pattern, replacement: replacement.replacement, to: output)
        }
        return output.lowercased()
    }

    private func replaceConsecutivePunctuation(_ text: String) -> String {
        var output = String()
        output.reserveCapacity(text.count)
        var previousWasPunctuation = false
        for character in text {
            let isPunctuation = punctuationCharacters.contains(character)
            if isPunctuation && previousWasPunctuation {
                continue
            }
            output.append(character)
            previousWasPunctuation = isPunctuation
        }
        return output
    }

    private func matchesJapaneseCharacters(_ text: String) -> Bool {
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        guard let match = japaneseCharacterRegex.firstMatch(in: text, options: [], range: range) else {
            return false
        }
        return match.range.location == 0
    }

    private func preferredNormText(for feature: JapaneseOpenJTalk.Feature) -> String {
        let value = !feature.pron.isEmpty && feature.pron != "*" ? feature.pron
            : (!feature.read.isEmpty && feature.read != "*" ? feature.read : feature.string)
        return value.replacingOccurrences(of: "’", with: "")
    }

    private func normalizePhone(_ phone: String) -> String {
        let replaced = postReplacePhone(phone)
        if GPTSoVITSPhoneSymbolAssets.symbolToID[replaced] != nil {
            return replaced
        }
        return "UNK"
    }

    private func postReplacePhone(_ phone: String) -> String {
        bundle.runtimeAssets.projectAssets.postReplaceMap[phone] ?? phone
    }

    private func phonesEquivalent(_ lhs: String, _ rhs: String) -> Bool {
        normalizeAlignmentPhone(lhs) == normalizeAlignmentPhone(rhs)
    }

    private func normalizeAlignmentPhone(_ phone: String) -> String {
        ["A", "E", "I", "O", "U"].contains(phone) ? phone.lowercased() : phone
    }

    private func isProsodyMark(_ phone: String) -> Bool {
        prosodyMarks.contains(phone)
    }

    private func numericFeature(_ pattern: String, in text: String) -> Int {
        guard let value = capture(pattern, in: text), let integer = Int(value) else {
            return -50
        }
        return integer
    }

    private func capture(_ pattern: String, in text: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return nil
        }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        guard let match = regex.firstMatch(in: text, options: [], range: range), match.numberOfRanges >= 2,
              let captureRange = Range(match.range(at: 1), in: text) else {
            return nil
        }
        return String(text[captureRange])
    }

    private func applyRegexReplacement(_ pattern: String, replacement: String, to text: String) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return text
        }
        return regex.stringByReplacingMatches(
            in: text,
            options: [],
            range: NSRange(text.startIndex..<text.endIndex, in: text),
            withTemplate: replacement
        )
    }
}
