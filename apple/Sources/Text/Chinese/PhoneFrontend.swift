import Foundation
import NaturalLanguage

public enum GPTSoVITSPhoneUnitType: String {
    case word
    case char
    case punct
}

public struct GPTSoVITSPhoneUnit {
    public let unitType: GPTSoVITSPhoneUnitType
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

public struct GPTSoVITSChinesePhoneResult {
    public let sourceText: String
    public let normalizedText: String
    public let phones: [String]
    public let phoneIDs: [Int]
    public let word2ph: [Int]
    public let phoneUnits: [GPTSoVITSPhoneUnit]
    public let backend: String
    public let unresolvedCharacters: [String]
}

public struct GPTSoVITSChineseSegmentPhoneResult {
    public let segment: GPTSoVITSTextSegment
    public let phoneResult: GPTSoVITSChinesePhoneResult
}

public struct GPTSoVITSChinesePhonePreprocessResult {
    public let preprocessResult: GPTSoVITSTextPreprocessResult
    public let segmentResults: [GPTSoVITSChineseSegmentPhoneResult]
}

public struct G2PWPredictionResult {
    public let normalizedText: String
    public let bopomofoByChar: [String?]
    public let phoneTemplateByChar: [GPTSoVITSPhoneSyllableTemplate?]
    public let queryCount: Int
    public let g2pwResolvedCount: Int
}

private struct GPTSoVITSLexiconSegmentResult {
    let units: [GPTSoVITSChineseLexicalUnitDraft]
    let trailingSingleCharBuffer: Bool
}

public struct GPTSoVITSPhoneSyllableTemplate {
    public let label: String
    public let initialPhone: String
    public let finalPhoneBase: String
    public let tone: Int
}

public struct GPTSoVITSPossegHMM {
    public let charStateTab: [String: [String]]
    public let startProb: [String: Double]
    public let transProb: [String: [String: Double]]
    public let emitProb: [String: [String: Double]]
}

public struct GPTSoVITSChineseFrontendLexicon {
    public struct PhraseUnitBreakdown {
        public let text: String
        public let pos: String
        public let charStart: Int
        public let charEnd: Int
    }

    public let wordFrequency: [String: Int]
    public let wordPOS: [String: String]
    public let forcedWords: Set<String>
    public let possegHMM: GPTSoVITSPossegHMM?
    public let totalFrequency: Double
    public let maxWordLength: Int
    public let phraseTemplates: [String: [GPTSoVITSPhoneSyllableTemplate]]
    public let phraseUnitBreakdowns: [String: [PhraseUnitBreakdown]]
    public let mustNeutralToneWords: Set<String>
    public let mustNotNeutralToneWords: Set<String>
    public let mustErhuaWords: Set<String>
    public let notErhuaWords: Set<String>
    public let punctuation: Set<Character>
    public let traditionalToSimplifiedMap: [String: String]
}

public protocol GPTSoVITSG2PWPredicting {
    var frontendLexicon: GPTSoVITSChineseFrontendLexicon? { get }
    func predictSyllableAlignment(for normalizedText: String) throws -> G2PWPredictionResult
}

public enum GPTSoVITSChinesePhoneFrontendError: LocalizedError {
    case unsupportedLanguage(String)

    public var errorDescription: String? {
        switch self {
        case let .unsupportedLanguage(language):
            return "GPTSoVITSChinesePhoneFrontend 当前只支持中文主链，收到 language=\(language)。"
        }
    }
}

public extension GPTSoVITSTextFrontend {
    func preprocessChinesePhoneSegments(
        text: String,
        language: GPTSoVITSTextLanguage = .zh,
        splitMethod: GPTSoVITSTextSplitMethod = .cut5,
        g2pwDriver: (any GPTSoVITSG2PWPredicting)? = nil
    ) throws -> GPTSoVITSChinesePhonePreprocessResult {
        guard language.baseLanguage == "zh" else {
            throw GPTSoVITSChinesePhoneFrontendError.unsupportedLanguage(language.rawValue)
        }
        let preprocessResult = preprocess(text: text, language: language, splitMethod: splitMethod)
        let segmentResults = try preprocessResult.segments.map { segment in
            GPTSoVITSChineseSegmentPhoneResult(
                segment: segment,
                phoneResult: try chinesePhoneResult(for: segment.text, language: language, g2pwDriver: g2pwDriver)
            )
        }
        return GPTSoVITSChinesePhonePreprocessResult(
            preprocessResult: preprocessResult,
            segmentResults: segmentResults
        )
    }

    func chinesePhoneResult(
        for text: String,
        language: GPTSoVITSTextLanguage = .zh,
        g2pwDriver: (any GPTSoVITSG2PWPredicting)? = nil
    ) throws -> GPTSoVITSChinesePhoneResult {
        guard language.baseLanguage == "zh" else {
            throw GPTSoVITSChinesePhoneFrontendError.unsupportedLanguage(language.rawValue)
        }

        let normalizedText = normalizeInputText(
            text,
            language: language,
            traditionalToSimplifiedMap: g2pwDriver?.frontendLexicon?.traditionalToSimplifiedMap
        )
        return try Self.buildChinesePhoneResult(
            sourceText: text,
            normalizedText: normalizedText,
            g2pwDriver: g2pwDriver
        )
    }

    private static func buildChinesePhoneResult(
        sourceText: String,
        normalizedText: String,
        g2pwDriver: (any GPTSoVITSG2PWPredicting)?
    ) throws -> GPTSoVITSChinesePhoneResult {
        let unkID = GPTSoVITSPhoneSymbolAssets.symbolToID["UNK"] ?? 0
        let g2pwPrediction = try g2pwDriver?.predictSyllableAlignment(for: normalizedText)
        let alignedTemplates = g2pwPrediction?.phoneTemplateByChar
        let lexicon = g2pwDriver?.frontendLexicon
        let lexicalUnits = buildLexicalUnits(
            from: normalizedText,
            alignedTemplates: alignedTemplates,
            lexicon: lexicon
        )
        var unresolvedCharacters = [String]()
        unresolvedCharacters.reserveCapacity(normalizedText.count)
        let normalizedCharacters = Array(normalizedText)
        var phones = [String]()
        var phoneIDs = [Int]()
        var word2ph = Array(repeating: 0, count: normalizedCharacters.count)
        var phoneUnits = [GPTSoVITSPhoneUnit]()
        var phoneCursor = 0

        for (unitIndex, unit) in lexicalUnits.enumerated() {
            var unitPhones = [String]()
            var unitPhoneIDs = [Int]()

            switch unit.unitType {
            case .punct:
                let phone = unit.text
                unitPhones = [phone]
                unitPhoneIDs = [GPTSoVITSPhoneSymbolAssets.symbolToID[phone] ?? unkID]
                if unit.charStart < word2ph.count {
                    word2ph[unit.charStart] = 1
                }
            case .word, .char:
                let drafts = buildSyllableDrafts(
                    for: unit,
                    unitIndex: unitIndex,
                    alignedTemplates: alignedTemplates,
                    lexicon: lexicon,
                    unresolvedCharacters: &unresolvedCharacters
                )
                for draft in drafts {
                    let mappedPhones = mapSyllableDraftToPhones(draft)
                    let mappedPhoneIDs = mappedPhones.map { GPTSoVITSPhoneSymbolAssets.symbolToID[$0] ?? unkID }
                    unitPhones.append(contentsOf: mappedPhones)
                    unitPhoneIDs.append(contentsOf: mappedPhoneIDs)
                    if draft.globalCharIndex < word2ph.count {
                        word2ph[draft.globalCharIndex] = mappedPhones.count
                    }
                }
            }

            phones.append(contentsOf: unitPhones)
            phoneIDs.append(contentsOf: unitPhoneIDs)
            phoneUnits.append(
                GPTSoVITSPhoneUnit(
                    unitType: unit.unitType,
                    text: unit.text,
                    normText: unit.text,
                    pos: unit.pos,
                    phones: unitPhones,
                    phoneIDs: unitPhoneIDs,
                    charStart: unit.charStart,
                    charEnd: unit.charEnd,
                    phoneStart: phoneCursor,
                    phoneEnd: phoneCursor + unitPhones.count,
                    phoneCount: unitPhones.count
                )
            )
            phoneCursor += unitPhones.count
        }

        return GPTSoVITSChinesePhoneResult(
            sourceText: sourceText,
            normalizedText: normalizedText,
            phones: phones,
            phoneIDs: phoneIDs,
            word2ph: word2ph,
            phoneUnits: phoneUnits,
            backend: g2pwPrediction?.g2pwResolvedCount ?? 0 > 0
                ? "apple_g2pw_coreml_plus_native_tone_sandhi"
                : "apple_native_word_context_tone_sandhi",
            unresolvedCharacters: unresolvedCharacters
        )
    }

    private static func buildSyllableDrafts(
        for unit: GPTSoVITSChineseLexicalUnitDraft,
        unitIndex: Int,
        alignedTemplates: [GPTSoVITSPhoneSyllableTemplate?]?,
        lexicon: GPTSoVITSChineseFrontendLexicon?,
        unresolvedCharacters: inout [String]
    ) -> [GPTSoVITSChineseSyllableDraft] {
        let characters = Array(unit.text)
        let hasAuthoritativePhraseTemplate = (lexicon?.phraseTemplates[unit.text]?.count == characters.count)
        let baseTemplates = resolveBaseTemplates(
            for: unit,
            alignedTemplates: alignedTemplates,
            lexicon: lexicon
        )
        var drafts = [GPTSoVITSChineseSyllableDraft]()
        drafts.reserveCapacity(characters.count)

        for (offset, character) in characters.enumerated() {
            let globalCharIndex = unit.charStart + offset
            let resolvedTemplate = offset < baseTemplates.count ? baseTemplates[offset] : nil
            if let template = resolvedTemplate {
                drafts.append(
                    GPTSoVITSChineseSyllableDraft(
                        character: character,
                        unitIndex: unitIndex,
                        segmentIndex: unit.segmentIndex,
                        globalCharIndex: globalCharIndex,
                        literalPhone: nil,
                        initialPhone: template.initialPhone,
                        finalPhoneBase: template.finalPhoneBase,
                        tone: template.tone
                    )
                )
            } else if isPhonePunctuation(character, lexicon: lexicon) {
                drafts.append(
                    GPTSoVITSChineseSyllableDraft(
                        character: character,
                        unitIndex: unitIndex,
                        segmentIndex: unit.segmentIndex,
                        globalCharIndex: globalCharIndex,
                        literalPhone: String(character),
                        initialPhone: nil,
                        finalPhoneBase: nil,
                        tone: 0
                    )
                )
            } else {
                unresolvedCharacters.append(String(character))
                drafts.append(
                    GPTSoVITSChineseSyllableDraft(
                        character: character,
                        unitIndex: unitIndex,
                        segmentIndex: unit.segmentIndex,
                        globalCharIndex: globalCharIndex,
                        literalPhone: nil,
                        initialPhone: nil,
                        finalPhoneBase: nil,
                        tone: 5
                    )
                )
            }
        }

        if !hasAuthoritativePhraseTemplate {
            applyModifiedTone(to: &drafts, word: unit.text, pos: unit.pos ?? "x", lexicon: lexicon)
            applyErhua(to: &drafts, word: unit.text, pos: unit.pos ?? "x", lexicon: lexicon)
        }
        return drafts
    }

    private static func resolveBaseTemplates(
        for unit: GPTSoVITSChineseLexicalUnitDraft,
        alignedTemplates: [GPTSoVITSPhoneSyllableTemplate?]?,
        lexicon: GPTSoVITSChineseFrontendLexicon?
    ) -> [GPTSoVITSPhoneSyllableTemplate?] {
        let characters = Array(unit.text)
        if let phraseTemplates = lexicon?.phraseTemplates[unit.text], phraseTemplates.count == characters.count {
            return phraseTemplates.map(Optional.some)
        }

        let contextualSyllables = alignedTemplates == nil ? toneNumberSyllables(for: unit.text) : []
        return characters.enumerated().map { offset, character in
            let globalCharIndex = unit.charStart + offset
            if character == "儿" {
                return toneNumberPinyin(for: character).flatMap(phoneTemplate(from:))
            }
            if let alignedTemplates, globalCharIndex < alignedTemplates.count {
                return alignedTemplates[globalCharIndex] ??
                    toneNumberPinyin(for: character).flatMap(phoneTemplate(from:))
            }
            if contextualSyllables.count == characters.count {
                return phoneTemplate(from: contextualSyllables[offset])
            }
            return toneNumberPinyin(for: character).flatMap(phoneTemplate(from:))
        }
    }

    private static func buildLexicalUnits(
        from normalizedText: String,
        alignedTemplates: [GPTSoVITSPhoneSyllableTemplate?]?,
        lexicon: GPTSoVITSChineseFrontendLexicon?
    ) -> [GPTSoVITSChineseLexicalUnitDraft] {
        let characters = Array(normalizedText)
        guard !characters.isEmpty else { return [] }

        var rawUnits = [GPTSoVITSChineseLexicalUnitDraft]()
        var trailingSingleCharBuffer = false
        var cursor = 0
        while cursor < characters.count {
            let character = characters[cursor]
            if isStandalonePunctuationBoundary(character, lexicon: lexicon) {
                rawUnits.append(
                    GPTSoVITSChineseLexicalUnitDraft(
                        unitType: .punct,
                        text: String(character),
                        pos: character == "." && trailingSingleCharBuffer
                            ? "m"
                            : punctuationPOS(for: character, lexicon: lexicon),
                        charStart: cursor,
                        charEnd: cursor + 1,
                        segmentIndex: 0
                    )
                )
                trailingSingleCharBuffer = false
                cursor += 1
                continue
            }
            let spanStart = cursor
            while cursor < characters.count, !isStandalonePunctuationBoundary(characters[cursor], lexicon: lexicon) {
                cursor += 1
            }
            let spanText = String(characters[spanStart..<cursor])
            guard !spanText.isEmpty else { continue }

            let segmented: [GPTSoVITSChineseLexicalUnitDraft]
            if let lexicon {
                let result = segmentSpanWithLexicon(spanText, globalStart: spanStart, lexicon: lexicon)
                trailingSingleCharBuffer = result.trailingSingleCharBuffer
                segmented = result.units
            } else {
                trailingSingleCharBuffer = false
                segmented = segmentSpanWithTokenizer(spanText, globalStart: spanStart)
            }
            rawUnits.append(contentsOf: segmented)
        }
        let mergedForcedUnits = if let lexicon {
            mergeForcedWords(rawUnits, lexicon: lexicon)
        } else {
            rawUnits
        }
        let mergedUnits = applyPreMergeForModify(
            mergedForcedUnits,
            alignedTemplates: alignedTemplates,
            lexicon: lexicon
        )
        return assignSegmentIndices(resolveUnitPOS(mergedUnits, lexicon: lexicon))
    }

    private static func resolveUnitPOS(
        _ units: [GPTSoVITSChineseLexicalUnitDraft],
        lexicon: GPTSoVITSChineseFrontendLexicon?
    ) -> [GPTSoVITSChineseLexicalUnitDraft] {
        guard let lexicon else {
            return units
        }
        return units.map { unit in
            if unit.unitType == .punct {
                return GPTSoVITSChineseLexicalUnitDraft(
                    unitType: unit.unitType,
                    text: unit.text,
                    pos: unit.pos,
                    charStart: unit.charStart,
                    charEnd: unit.charEnd,
                    segmentIndex: unit.segmentIndex
                )
            }
            if (unit.pos == nil || unit.pos == "x"),
               let resolvedPOS = lexicon.wordPOS[unit.text] {
                return GPTSoVITSChineseLexicalUnitDraft(
                    unitType: unit.unitType,
                    text: unit.text,
                    pos: resolvedPOS,
                    charStart: unit.charStart,
                    charEnd: unit.charEnd,
                    segmentIndex: unit.segmentIndex
                )
            }
            return unit
        }
    }

    private static func punctuationPOS(
        for character: Character,
        lexicon: GPTSoVITSChineseFrontendLexicon?
    ) -> String {
        return lexicon?.wordPOS[String(character)] ?? "x"
    }

    private static func isStandalonePunctuationBoundary(
        _ character: Character,
        lexicon: GPTSoVITSChineseFrontendLexicon?
    ) -> Bool {
        if shouldAttachToInternalLexiconBlock(character, lexicon: lexicon) {
            return false
        }
        return isPhonePunctuation(character, lexicon: lexicon)
    }

    private static func shouldAttachToInternalLexiconBlock(
        _ character: Character,
        lexicon: GPTSoVITSChineseFrontendLexicon?
    ) -> Bool {
        guard lexicon != nil else {
            return false
        }
        return jiebaInternalPunctuation.contains(character)
    }

    private static func isDetailNumericCharacter(_ character: Character) -> Bool {
        character == "." || character.isWholeNumber
    }

    private static func isDetailEnglishCharacter(_ character: Character) -> Bool {
        character.unicodeScalars.allSatisfy { scalar in
            scalar.isASCII && CharacterSet.alphanumerics.contains(scalar)
        }
    }

    private static func assignSegmentIndices(
        _ units: [GPTSoVITSChineseLexicalUnitDraft]
    ) -> [GPTSoVITSChineseLexicalUnitDraft] {
        var finalized = [GPTSoVITSChineseLexicalUnitDraft]()
        finalized.reserveCapacity(units.count)
        var segmentIndex = 0
        for unit in units {
            finalized.append(
                GPTSoVITSChineseLexicalUnitDraft(
                    unitType: unit.unitType,
                    text: unit.text,
                    pos: unit.pos,
                    charStart: unit.charStart,
                    charEnd: unit.charEnd,
                    segmentIndex: segmentIndex
                )
            )
            if unit.unitType == .punct {
                segmentIndex += 1
            }
        }
        return finalized
    }

    private static func segmentSpanWithTokenizer(
        _ text: String,
        globalStart: Int
    ) -> [GPTSoVITSChineseLexicalUnitDraft] {
        let characters = Array(text)
        guard !characters.isEmpty else { return [] }
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text
        var units = [GPTSoVITSChineseLexicalUnitDraft]()
        var occupied = Array(repeating: false, count: characters.count)

        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let start = text.distance(from: text.startIndex, to: range.lowerBound)
            let end = text.distance(from: text.startIndex, to: range.upperBound)
            let token = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !token.isEmpty else { return true }
            units.append(
                GPTSoVITSChineseLexicalUnitDraft(
                    unitType: .word,
                    text: token,
                    pos: "x",
                    charStart: globalStart + start,
                    charEnd: globalStart + end,
                    segmentIndex: 0
                )
            )
            for index in start..<end where index < occupied.count {
                occupied[index] = true
            }
            return true
        }

        for index in characters.indices where !occupied[index] {
            units.append(
                GPTSoVITSChineseLexicalUnitDraft(
                    unitType: .word,
                    text: String(characters[index]),
                    pos: "x",
                    charStart: globalStart + index,
                    charEnd: globalStart + index + 1,
                    segmentIndex: 0
                )
            )
        }
        return units.sorted { $0.charStart < $1.charStart }
    }

    private static func segmentSpanWithLexicon(
        _ text: String,
        globalStart: Int,
        lexicon: GPTSoVITSChineseFrontendLexicon
    ) -> GPTSoVITSLexiconSegmentResult {
        let characters = Array(text)
        guard !characters.isEmpty else {
            return GPTSoVITSLexiconSegmentResult(units: [], trailingSingleCharBuffer: false)
        }
        if let authoritativeUnits = authoritativePhraseUnits(
            for: text,
            globalStart: globalStart,
            lexicon: lexicon
        ) {
            return GPTSoVITSLexiconSegmentResult(units: authoritativeUnits, trailingSingleCharBuffer: false)
        }

        struct Route {
            let score: Double
            let nextIndex: Int
        }

        let logTotal = log(max(lexicon.totalFrequency, 1.0))
        var route = Array(
            repeating: Route(score: -.infinity, nextIndex: text.count),
            count: characters.count + 1
        )
        route[characters.count] = Route(score: 0, nextIndex: characters.count)

        for start in stride(from: characters.count - 1, through: 0, by: -1) {
            var bestScore = -Double.infinity
            var bestEnd = start + 1
            let maxLength = min(lexicon.maxWordLength, characters.count - start)
            if maxLength >= 1 {
                for length in 1...maxLength {
                    let end = start + length
                    let word = String(characters[start..<end])
                    if length > 1, word.contains(where: { isPhonePunctuation($0, lexicon: lexicon) }) {
                        continue
                    }
                    let fallbackFrequency = length == 1 ? 1 : 0
                    let frequency = max(lexicon.wordFrequency[word] ?? fallbackFrequency, 0)
                    guard frequency > 0 else { continue }
                    let score = log(Double(frequency)) - logTotal + route[end].score
                    if score > bestScore {
                        bestScore = score
                        bestEnd = end
                    }
                }
            }
            route[start] = Route(score: bestScore, nextIndex: bestEnd)
        }

        var units = [GPTSoVITSChineseLexicalUnitDraft]()
        var cursor = 0
        while cursor < characters.count {
            let next = max(route[cursor].nextIndex, cursor + 1)
            let word = String(characters[cursor..<next])
            let unitType: GPTSoVITSPhoneUnitType
            let pos: String
            if next == cursor + 1, let token = word.first, isPhonePunctuation(token, lexicon: lexicon) {
                unitType = .punct
                pos = punctuationPOS(for: token, lexicon: lexicon)
            } else {
                unitType = .word
                pos = lexicon.wordPOS[word] ?? "x"
            }
            units.append(
                GPTSoVITSChineseLexicalUnitDraft(
                    unitType: unitType,
                    text: word,
                    pos: pos,
                    charStart: globalStart + cursor,
                    charEnd: globalStart + next,
                    segmentIndex: 0
                )
            )
            cursor = next
        }
        let trailingSingleCharBuffer = trailingDictionaryBuffer(units, lexicon: lexicon)
        return GPTSoVITSLexiconSegmentResult(
            units: applyPossegHMMFallback(units, lexicon: lexicon),
            trailingSingleCharBuffer: trailingSingleCharBuffer
        )
    }

    private static func authoritativePhraseUnits(
        for text: String,
        globalStart: Int,
        lexicon: GPTSoVITSChineseFrontendLexicon
    ) -> [GPTSoVITSChineseLexicalUnitDraft]? {
        let characters = Array(text)
        var suffixPunctuation = [Character]()
        var coreCharacters = characters
        while let last = coreCharacters.last, jiebaInternalPunctuation.contains(last) {
            suffixPunctuation.insert(last, at: 0)
            coreCharacters.removeLast()
        }
        let coreText = String(coreCharacters)
        guard !coreText.isEmpty,
              let breakdowns = lexicon.phraseUnitBreakdowns[coreText]
        else {
            return nil
        }

        var units = [GPTSoVITSChineseLexicalUnitDraft]()
        units.reserveCapacity(breakdowns.count + suffixPunctuation.count)
        var appendedSuffixCount = 0
        for breakdown in breakdowns {
            if breakdown.charStart >= coreText.count {
                let suffixIndex = breakdown.charStart - coreText.count
                guard suffixIndex < suffixPunctuation.count,
                      breakdown.text == String(suffixPunctuation[suffixIndex])
                else {
                    continue
                }
                units.append(
                    GPTSoVITSChineseLexicalUnitDraft(
                        unitType: .punct,
                        text: breakdown.text,
                        pos: breakdown.pos,
                        charStart: globalStart + breakdown.charStart,
                        charEnd: globalStart + breakdown.charEnd,
                        segmentIndex: 0
                    )
                )
                appendedSuffixCount += 1
            } else {
                units.append(
                    GPTSoVITSChineseLexicalUnitDraft(
                        unitType: .word,
                        text: breakdown.text,
                        pos: breakdown.pos,
                        charStart: globalStart + breakdown.charStart,
                        charEnd: globalStart + breakdown.charEnd,
                        segmentIndex: 0
                    )
                )
            }
        }

        var punctuationStart = coreText.count + appendedSuffixCount
        for punctuation in suffixPunctuation.dropFirst(appendedSuffixCount) {
            units.append(
                GPTSoVITSChineseLexicalUnitDraft(
                    unitType: .punct,
                    text: String(punctuation),
                    pos: punctuationPOS(for: punctuation, lexicon: lexicon),
                    charStart: globalStart + punctuationStart,
                    charEnd: globalStart + punctuationStart + 1,
                    segmentIndex: 0
                )
            )
            punctuationStart += 1
        }
        return units
    }

    private static func trailingDictionaryBuffer(
        _ units: [GPTSoVITSChineseLexicalUnitDraft],
        lexicon: GPTSoVITSChineseFrontendLexicon
    ) -> Bool {
        guard !units.isEmpty else {
            return false
        }
        var buffer = [GPTSoVITSChineseLexicalUnitDraft]()
        for unit in units.reversed() {
            if unit.unitType == .punct {
                break
            }
            if unit.text.count == 1 {
                buffer.insert(unit, at: 0)
            } else {
                break
            }
        }
        guard buffer.count > 1 else {
            return false
        }
        let bufferText = buffer.map(\.text).joined()
        return (lexicon.wordFrequency[bufferText] ?? 0) <= 0
    }

    private static func applyPossegHMMFallback(
        _ units: [GPTSoVITSChineseLexicalUnitDraft],
        lexicon: GPTSoVITSChineseFrontendLexicon
    ) -> [GPTSoVITSChineseLexicalUnitDraft] {
        guard !units.isEmpty, lexicon.possegHMM != nil else {
            return units
        }

        var resolved = [GPTSoVITSChineseLexicalUnitDraft]()
        var buffer = [GPTSoVITSChineseLexicalUnitDraft]()

        func flushBuffer() {
            guard !buffer.isEmpty else {
                return
            }
            defer { buffer.removeAll(keepingCapacity: true) }
            if buffer.count == 1 {
                resolved.append(buffer[0])
                return
            }

            let bufferText = buffer.map(\.text).joined()
            if let frequency = lexicon.wordFrequency[bufferText], frequency > 0 {
                resolved.append(contentsOf: buffer)
                return
            }

            if let detailUnits = detailSegment(bufferText, globalStart: buffer[0].charStart, lexicon: lexicon) {
                resolved.append(contentsOf: detailUnits)
            } else {
                resolved.append(contentsOf: buffer)
            }
        }

        for unit in units {
            if unit.text.count == 1 {
                buffer.append(unit)
            } else {
                flushBuffer()
                resolved.append(unit)
            }
        }
        flushBuffer()
        return resolved
    }

    private static func detailSegment(
        _ text: String,
        globalStart: Int,
        lexicon: GPTSoVITSChineseFrontendLexicon
    ) -> [GPTSoVITSChineseLexicalUnitDraft]? {
        let characters = Array(text)
        guard !characters.isEmpty else {
            return nil
        }

        var units = [GPTSoVITSChineseLexicalUnitDraft]()
        var cursor = 0
        while cursor < characters.count {
            let character = characters[cursor]
            let blockStart = cursor
            if isChineseScalarCharacter(character) {
                while cursor < characters.count, isChineseScalarCharacter(characters[cursor]) {
                    cursor += 1
                }
                let blockText = String(characters[blockStart..<cursor])
                if let hmmUnits = hmmSegment(blockText, globalStart: globalStart + blockStart, lexicon: lexicon) {
                    units.append(contentsOf: hmmUnits)
                } else {
                    for index in blockStart..<cursor {
                        let token = String(characters[index])
                        units.append(
                            GPTSoVITSChineseLexicalUnitDraft(
                                unitType: .word,
                                text: token,
                                pos: lexicon.wordPOS[token] ?? "x",
                                charStart: globalStart + index,
                                charEnd: globalStart + index + 1,
                                segmentIndex: 0
                            )
                        )
                    }
                }
                continue
            }

            if isDetailNumericCharacter(character) {
                while cursor < characters.count, isDetailNumericCharacter(characters[cursor]) {
                    cursor += 1
                }
                let blockText = String(characters[blockStart..<cursor])
                let unitType: GPTSoVITSPhoneUnitType = blockText == "." ? .punct : .word
                units.append(
                    GPTSoVITSChineseLexicalUnitDraft(
                        unitType: unitType,
                        text: blockText,
                        pos: "m",
                        charStart: globalStart + blockStart,
                        charEnd: globalStart + cursor,
                        segmentIndex: 0
                    )
                )
                continue
            }

            if isDetailEnglishCharacter(character) {
                while cursor < characters.count, isDetailEnglishCharacter(characters[cursor]) {
                    cursor += 1
                }
                let blockText = String(characters[blockStart..<cursor])
                units.append(
                    GPTSoVITSChineseLexicalUnitDraft(
                        unitType: .word,
                        text: blockText,
                        pos: "eng",
                        charStart: globalStart + blockStart,
                        charEnd: globalStart + cursor,
                        segmentIndex: 0
                    )
                )
                continue
            }

            cursor += 1
            units.append(
                GPTSoVITSChineseLexicalUnitDraft(
                    unitType: isPhonePunctuation(character, lexicon: lexicon) ? .punct : .word,
                    text: String(character),
                    pos: isPhonePunctuation(character, lexicon: lexicon)
                        ? punctuationPOS(for: character, lexicon: lexicon)
                        : "x",
                    charStart: globalStart + blockStart,
                    charEnd: globalStart + cursor,
                    segmentIndex: 0
                )
            )
        }
        return units
    }

    private static func hmmSegment(
        _ text: String,
        globalStart: Int,
        lexicon: GPTSoVITSChineseFrontendLexicon
    ) -> [GPTSoVITSChineseLexicalUnitDraft]? {
        guard let hmm = lexicon.possegHMM else {
            return nil
        }
        let characters = Array(text).map(String.init)
        guard characters.count > 1 else {
            return nil
        }
        let allStates = Array(hmm.transProb.keys)
        guard !allStates.isEmpty else {
            return nil
        }

        var scores = [String: Double]()
        var backPointers = [[String: String]]()
        let firstStates = hmm.charStateTab[characters[0]] ?? allStates
        var firstBack = [String: String]()
        for state in firstStates {
            let emit = hmm.emitProb[state]?[characters[0]] ?? -3.14e100
            let start = hmm.startProb[state] ?? -3.14e100
            scores[state] = start + emit
            firstBack[state] = ""
        }
        guard !scores.isEmpty else {
            return nil
        }
        backPointers.append(firstBack)

        if characters.count > 1 {
            for index in 1..<characters.count {
                let character = characters[index]
                let previousScores = scores
                let previousStates = Array(previousScores.keys).filter {
                    !(hmm.transProb[$0]?.isEmpty ?? true)
                }
                guard !previousStates.isEmpty else {
                    return nil
                }
                let expectedStates = Set(previousStates.flatMap { previous in
                    if let nextKeys = hmm.transProb[previous]?.keys {
                        return Array(nextKeys)
                    }
                    return [String]()
                })
                var candidateStates = Set(hmm.charStateTab[character] ?? allStates).intersection(expectedStates)
                if candidateStates.isEmpty {
                    candidateStates = expectedStates.isEmpty ? Set(allStates) : expectedStates
                }
                var nextScores = [String: Double]()
                var nextBack = [String: String]()
                for state in candidateStates {
                    let emit = hmm.emitProb[state]?[character] ?? -3.14e100
                    var bestScore = -Double.infinity
                    var bestPrevious = ""
                    for previous in previousStates {
                        let transition = hmm.transProb[previous]?[state] ?? -Double.infinity
                        let score = (previousScores[previous] ?? -Double.infinity) + transition + emit
                        if score > bestScore {
                            bestScore = score
                            bestPrevious = previous
                        }
                    }
                    if !bestPrevious.isEmpty {
                        nextScores[state] = bestScore
                        nextBack[state] = bestPrevious
                    }
                }
                guard !nextScores.isEmpty else {
                    return nil
                }
                scores = nextScores
                backPointers.append(nextBack)
            }
        }

        guard let finalState = scores.max(by: { $0.value < $1.value })?.key else {
            return nil
        }

        var route = Array(repeating: "", count: characters.count)
        var state = finalState
        for index in stride(from: characters.count - 1, through: 0, by: -1) {
            route[index] = state
            state = backPointers[index][state] ?? ""
        }

        var units = [GPTSoVITSChineseLexicalUnitDraft]()
        var begin = 0
        var nextIndex = 0
        for (index, stateLabel) in route.enumerated() {
            let components = stateLabel.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false)
            guard let marker = components.first?.first else {
                continue
            }
            let pos = components.count > 1 ? String(components[1]) : "x"
            switch marker {
            case "B":
                begin = index
            case "E":
                let token = characters[begin...index].joined()
                units.append(
                    GPTSoVITSChineseLexicalUnitDraft(
                        unitType: .word,
                        text: token,
                        pos: pos,
                        charStart: globalStart + begin,
                        charEnd: globalStart + index + 1,
                        segmentIndex: 0
                    )
                )
                nextIndex = index + 1
            case "S":
                let token = characters[index]
                units.append(
                    GPTSoVITSChineseLexicalUnitDraft(
                        unitType: .word,
                        text: token,
                        pos: pos,
                        charStart: globalStart + index,
                        charEnd: globalStart + index + 1,
                        segmentIndex: 0
                    )
                )
                nextIndex = index + 1
            default:
                continue
            }
        }

        if nextIndex < characters.count {
            let fallbackState = route[nextIndex]
            let components = fallbackState.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false)
            let pos = components.count > 1 ? String(components[1]) : "x"
            units.append(
                GPTSoVITSChineseLexicalUnitDraft(
                    unitType: .word,
                    text: characters[nextIndex...].joined(),
                    pos: pos,
                    charStart: globalStart + nextIndex,
                    charEnd: globalStart + characters.count,
                    segmentIndex: 0
                )
            )
        }

        guard !units.isEmpty else {
            return nil
        }
        return units
    }

    private static func mergeForcedWords(
        _ units: [GPTSoVITSChineseLexicalUnitDraft],
        lexicon: GPTSoVITSChineseFrontendLexicon
    ) -> [GPTSoVITSChineseLexicalUnitDraft] {
        guard !units.isEmpty, !lexicon.forcedWords.isEmpty else {
            return units
        }
        let maxForcedWordLength = lexicon.forcedWords.map(\.count).max() ?? 1
        guard maxForcedWordLength > 1 else {
            return units
        }

        var merged = [GPTSoVITSChineseLexicalUnitDraft]()
        var index = 0
        while index < units.count {
            var bestEndIndex: Int?
            var bestText: String?
            var combinedLength = 0
            var candidate = ""

            for endIndex in index..<units.count {
                let unit = units[endIndex]
                if unit.unitType == .punct {
                    break
                }
                combinedLength += unit.text.count
                if combinedLength > maxForcedWordLength {
                    break
                }
                candidate += unit.text
                if lexicon.forcedWords.contains(candidate) {
                    bestEndIndex = endIndex
                    bestText = candidate
                }
            }

            if let bestEndIndex,
               let bestText,
               bestEndIndex > index {
                let first = units[index]
                let last = units[bestEndIndex]
                merged.append(
                    GPTSoVITSChineseLexicalUnitDraft(
                        unitType: .word,
                        text: bestText,
                        pos: lexicon.wordPOS[bestText] ?? first.pos ?? last.pos,
                        charStart: first.charStart,
                        charEnd: last.charEnd,
                        segmentIndex: 0
                    )
                )
                index = bestEndIndex + 1
                continue
            }

            merged.append(units[index])
            index += 1
        }
        return merged
    }

    private static func applyPreMergeForModify(
        _ units: [GPTSoVITSChineseLexicalUnitDraft],
        alignedTemplates: [GPTSoVITSPhoneSyllableTemplate?]?,
        lexicon: GPTSoVITSChineseFrontendLexicon?
    ) -> [GPTSoVITSChineseLexicalUnitDraft] {
        guard !units.isEmpty else { return [] }
        var merged = mergeBu(units)
        merged = mergeYi(merged)
        merged = mergeReduplication(merged)
        merged = mergeContinuousThreeTones(
            merged,
            alignedTemplates: alignedTemplates,
            lexicon: lexicon
        )
        merged = mergeContinuousThreeTones2(
            merged,
            alignedTemplates: alignedTemplates,
            lexicon: lexicon
        )
        merged = mergeEr(merged)
        return merged
    }

    private static func mergeBu(
        _ units: [GPTSoVITSChineseLexicalUnitDraft]
    ) -> [GPTSoVITSChineseLexicalUnitDraft] {
        var merged = [GPTSoVITSChineseLexicalUnitDraft]()
        var pendingBu: GPTSoVITSChineseLexicalUnitDraft?
        for unit in units {
            if let pending = pendingBu {
                merged.append(mergeLexicalUnits(pending, unit, text: pending.text + unit.text, pos: unit.pos))
                pendingBu = nil
                continue
            }
            if unit.text == "不" {
                pendingBu = unit
            } else {
                merged.append(unit)
            }
        }
        if let pendingBu {
            merged.append(
                GPTSoVITSChineseLexicalUnitDraft(
                    unitType: .word,
                    text: pendingBu.text,
                    pos: "d",
                    charStart: pendingBu.charStart,
                    charEnd: pendingBu.charEnd,
                    segmentIndex: 0
                )
            )
        }
        return merged
    }

    private static func mergeYi(
        _ units: [GPTSoVITSChineseLexicalUnitDraft]
    ) -> [GPTSoVITSChineseLexicalUnitDraft] {
        var stageOne = [GPTSoVITSChineseLexicalUnitDraft]()
        var index = 0
        while index < units.count {
            let unit = units[index]
            var didMerge = false
            if index > 0, unit.text == "一", index + 1 < units.count, !stageOne.isEmpty {
                let previous = stageOne[stageOne.count - 1]
                let next = units[index + 1]
                if previous.text == next.text, previous.pos == "v", next.pos == "v" {
                    stageOne[stageOne.count - 1] = mergeLexicalUnits(
                        previous,
                        next,
                        text: previous.text + unit.text + next.text,
                        pos: previous.pos
                    )
                    index += 2
                    didMerge = true
                }
            }
            if !didMerge {
                stageOne.append(unit)
                index += 1
            }
        }

        var stageTwo = [GPTSoVITSChineseLexicalUnitDraft]()
        for unit in stageOne {
            if let previous = stageTwo.last, previous.text == "一" {
                stageTwo[stageTwo.count - 1] = mergeLexicalUnits(
                    previous,
                    unit,
                    text: previous.text + unit.text,
                    pos: previous.pos
                )
            } else {
                stageTwo.append(unit)
            }
        }
        return stageTwo
    }

    private static func mergeReduplication(
        _ units: [GPTSoVITSChineseLexicalUnitDraft]
    ) -> [GPTSoVITSChineseLexicalUnitDraft] {
        var merged = [GPTSoVITSChineseLexicalUnitDraft]()
        for unit in units {
            if let previous = merged.last, previous.text == unit.text {
                merged[merged.count - 1] = mergeLexicalUnits(previous, unit)
            } else {
                merged.append(unit)
            }
        }
        return merged
    }

    private static func mergeContinuousThreeTones(
        _ units: [GPTSoVITSChineseLexicalUnitDraft],
        alignedTemplates: [GPTSoVITSPhoneSyllableTemplate?]?,
        lexicon: GPTSoVITSChineseFrontendLexicon?
    ) -> [GPTSoVITSChineseLexicalUnitDraft] {
        let toneLists = units.map { baseTones(for: $0, alignedTemplates: alignedTemplates, lexicon: lexicon) }
        var merged = [GPTSoVITSChineseLexicalUnitDraft]()
        var mergeLast = Array(repeating: false, count: units.count)
        for index in units.indices {
            if index > 0,
               let previousTones = toneLists[index - 1],
               let currentTones = toneLists[index],
               allToneThree(previousTones),
               allToneThree(currentTones),
               !mergeLast[index - 1]
            {
                let previous = units[index - 1]
                let current = units[index]
                if !isReduplication(previous.text), previous.text.count + current.text.count <= 3, !merged.isEmpty {
                    merged[merged.count - 1] = mergeLexicalUnits(merged[merged.count - 1], current)
                    mergeLast[index] = true
                } else {
                    merged.append(current)
                }
            } else {
                merged.append(units[index])
            }
        }
        return merged
    }

    private static func mergeContinuousThreeTones2(
        _ units: [GPTSoVITSChineseLexicalUnitDraft],
        alignedTemplates: [GPTSoVITSPhoneSyllableTemplate?]?,
        lexicon: GPTSoVITSChineseFrontendLexicon?
    ) -> [GPTSoVITSChineseLexicalUnitDraft] {
        let toneLists = units.map { baseTones(for: $0, alignedTemplates: alignedTemplates, lexicon: lexicon) }
        var merged = [GPTSoVITSChineseLexicalUnitDraft]()
        var mergeLast = Array(repeating: false, count: units.count)
        for index in units.indices {
            if index > 0,
               let previousTones = toneLists[index - 1],
               let currentTones = toneLists[index],
               previousTones.last == 3,
               currentTones.first == 3,
               !mergeLast[index - 1]
            {
                let previous = units[index - 1]
                let current = units[index]
                if !isReduplication(previous.text), previous.text.count + current.text.count <= 3, !merged.isEmpty {
                    merged[merged.count - 1] = mergeLexicalUnits(merged[merged.count - 1], current)
                    mergeLast[index] = true
                } else {
                    merged.append(current)
                }
            } else {
                merged.append(units[index])
            }
        }
        return merged
    }

    private static func mergeEr(
        _ units: [GPTSoVITSChineseLexicalUnitDraft]
    ) -> [GPTSoVITSChineseLexicalUnitDraft] {
        var merged = [GPTSoVITSChineseLexicalUnitDraft]()
        for unit in units {
            if let previous = merged.last, unit.text == "儿", previous.text != "#" {
                merged[merged.count - 1] = mergeLexicalUnits(previous, unit)
            } else {
                merged.append(unit)
            }
        }
        return merged
    }

    private static func mergeLexicalUnits(
        _ lhs: GPTSoVITSChineseLexicalUnitDraft,
        _ rhs: GPTSoVITSChineseLexicalUnitDraft,
        text: String? = nil,
        pos: String? = nil
    ) -> GPTSoVITSChineseLexicalUnitDraft {
        let mergedText = text ?? (lhs.text + rhs.text)
        return GPTSoVITSChineseLexicalUnitDraft(
            unitType: lhs.unitType == .punct && rhs.unitType == .punct ? .punct : .word,
            text: mergedText,
            pos: pos ?? lhs.pos ?? rhs.pos,
            charStart: lhs.charStart,
            charEnd: rhs.charEnd,
            segmentIndex: 0
        )
    }

    private static func baseTones(
        for unit: GPTSoVITSChineseLexicalUnitDraft,
        alignedTemplates: [GPTSoVITSPhoneSyllableTemplate?]?,
        lexicon: GPTSoVITSChineseFrontendLexicon?
    ) -> [Int]? {
        let templates = resolveBaseTemplates(for: unit, alignedTemplates: alignedTemplates, lexicon: lexicon)
        guard templates.count == unit.text.count else { return nil }
        var tones = [Int]()
        tones.reserveCapacity(templates.count)
        for template in templates {
            guard let template else { return nil }
            tones.append(template.tone)
        }
        return tones
    }

    private static func toneNumberSyllables(for text: String) -> [GPTSoVITSToneNumberSyllable] {
        let mutable = NSMutableString(string: text)
        CFStringTransform(mutable, nil, kCFStringTransformToLatin, false)
        return String(mutable)
            .lowercased()
            .split(separator: " ")
            .compactMap { toneNumberSyllable(from: String($0)) }
    }

    private static func toneNumberSyllable(from markedSyllable: String) -> GPTSoVITSToneNumberSyllable? {
        if let last = markedSyllable.last,
           let tone = Int(String(last)),
           (1...5).contains(tone) {
            let basePart = String(markedSyllable.dropLast())
            let normalizedBase = basePart
                .replacingOccurrences(of: "ü", with: "v")
                .lowercased()
            let filteredBase = normalizedBase.filter { character in
                ("a"..."z").contains(String(character))
            }
            if !filteredBase.isEmpty {
                return GPTSoVITSToneNumberSyllable(base: filteredBase, tone: tone)
            }
        }

        var base = ""
        var tone = 5
        var hasLetter = false
        for character in markedSyllable.precomposedStringWithCanonicalMapping {
            if let marked = toneMarkedVowels[character] {
                base.append(marked.base)
                tone = marked.tone
                hasLetter = true
                continue
            }
            switch character {
            case "ü":
                base.append("v")
                hasLetter = true
            case "a"..."z":
                base.append(character)
                hasLetter = true
            default:
                continue
            }
        }
        guard hasLetter else { return nil }
        return GPTSoVITSToneNumberSyllable(base: base, tone: tone)
    }

    private static func phoneTemplate(from markedSyllable: String) -> GPTSoVITSPhoneSyllableTemplate? {
        toneNumberSyllable(from: markedSyllable).flatMap(phoneTemplate(from:))
    }

    private static func applyModifiedTone(
        to drafts: inout [GPTSoVITSChineseSyllableDraft],
        word: String,
        pos: String,
        lexicon: GPTSoVITSChineseFrontendLexicon?
    ) {
        guard !drafts.isEmpty else { return }
        applyBuSandhi(to: &drafts, word: word)
        applyYiSandhi(to: &drafts, word: word, lexicon: lexicon)
        applyNeuralSandhi(to: &drafts, word: word, pos: pos, lexicon: lexicon)
        applyThreeSandhi(to: &drafts, word: word, lexicon: lexicon)
    }

    private static func applyBuSandhi(
        to drafts: inout [GPTSoVITSChineseSyllableDraft],
        word: String
    ) {
        let characters = Array(word)
        if drafts.count == 3, characters.count == 3, characters[1] == "不" {
            drafts[1].tone = 5
            return
        }
        for index in drafts.indices where characters[index] == "不" && index + 1 < drafts.count {
            if drafts[index + 1].tone == 4 {
                drafts[index].tone = 2
            }
        }
    }

    private static func applyYiSandhi(
        to drafts: inout [GPTSoVITSChineseSyllableDraft],
        word: String,
        lexicon: GPTSoVITSChineseFrontendLexicon?
    ) {
        let characters = Array(word)
        guard drafts.count == characters.count else { return }
        if characters.contains("一"),
           characters.filter({ $0 != "一" }).allSatisfy({ $0.isNumber || isChineseNumericCharacter($0) }) {
            return
        }
        if characters.count == 3, characters[1] == "一", characters[0] == characters[2] {
            drafts[1].tone = 5
            return
        }
        if word.hasPrefix("第一"), drafts.count > 1 {
            drafts[1].tone = 1
            return
        }
        let punctuation = punctuationCharacters(lexicon: lexicon)
        for index in drafts.indices where characters[index] == "一" && index + 1 < drafts.count {
            if drafts[index + 1].tone == 4 {
                drafts[index].tone = 2
            } else if !punctuation.contains(characters[index + 1]) {
                drafts[index].tone = 4
            }
        }
    }

    private static func applyNeuralSandhi(
        to drafts: inout [GPTSoVITSChineseSyllableDraft],
        word: String,
        pos: String,
        lexicon: GPTSoVITSChineseFrontendLexicon?
    ) {
        let characters = Array(word)
        guard drafts.count == characters.count else { return }
        let mustNeutral = lexicon?.mustNeutralToneWords ?? Set<String>()
        let mustNotNeutral = lexicon?.mustNotNeutralToneWords ?? Set<String>()

        if let posHead = pos.first, ["n", "v", "a"].contains(String(posHead)), !mustNotNeutral.contains(word) {
            for index in 1..<characters.count where characters[index] == characters[index - 1] {
                drafts[index].tone = 5
            }
        }

        if let last = characters.last {
            let geIndex = characters.firstIndex(of: "个")
            if "吧呢哈啊呐噻嘛吖嗨呐哦哒额滴哩哟喽啰耶喔诶".contains(last) {
                drafts[drafts.count - 1].tone = 5
            } else if "的地得".contains(last) {
                drafts[drafts.count - 1].tone = 5
            } else if characters.count == 1 && "了着过".contains(last) && ["ul", "uz", "ug"].contains(pos) {
                drafts[drafts.count - 1].tone = 5
            } else if characters.count > 1 && "们子".contains(last) && ["r", "n"].contains(pos) && !mustNotNeutral.contains(word) {
                drafts[drafts.count - 1].tone = 5
            } else if characters.count > 1 && "上下里".contains(last) && ["s", "l", "f"].contains(pos) {
                drafts[drafts.count - 1].tone = 5
            } else if characters.count > 1 && "来去".contains(last) && "上下进出回过起开".contains(characters[characters.count - 2]) {
                drafts[drafts.count - 1].tone = 5
            } else if let geIndex,
                      (geIndex >= 1 && (characters[geIndex - 1].isNumber || "几有两半多各整每做是".contains(characters[geIndex - 1])))
                        || word == "个" {
                drafts[geIndex].tone = 5
            } else if mustNeutral.contains(word) || mustNeutral.contains(String(word.suffix(2))) {
                drafts[drafts.count - 1].tone = 5
            }
        }

        let splitWords = splitWord(word, lexicon: lexicon)
        guard splitWords.count == 2 else { return }
        var offset = 0
        for subword in splitWords {
            let length = subword.count
            guard length > 0, offset + length <= drafts.count else { continue }
            if mustNeutral.contains(subword) || mustNeutral.contains(String(subword.suffix(2))) {
                drafts[offset + length - 1].tone = 5
            }
            offset += length
        }
    }

    private static func applyThreeSandhi(
        to drafts: inout [GPTSoVITSChineseSyllableDraft],
        word: String,
        lexicon: GPTSoVITSChineseFrontendLexicon?
    ) {
        let tones = drafts.map(\.tone)
        if drafts.count == 2, allToneThree(tones) {
            drafts[0].tone = 2
            return
        }
        if drafts.count == 4 {
            if allToneThree(Array(tones[..<2])) {
                drafts[0].tone = 2
            }
            if allToneThree(Array(tones[2...])) {
                drafts[2].tone = 2
            }
            return
        }
        if drafts.count != 3 {
            return
        }
        let splitWords = splitWord(word, lexicon: lexicon)
        if allToneThree(tones) {
            if splitWords.first?.count == 2 {
                drafts[0].tone = 2
                drafts[1].tone = 2
            } else if splitWords.first?.count == 1 {
                drafts[1].tone = 2
            }
            return
        }
        guard let firstCount = splitWords.first?.count, firstCount > 0, firstCount < drafts.count else {
            return
        }
        let firstTones = Array(tones[..<firstCount])
        let secondTones = Array(tones[firstCount...])
        if firstTones.count == 2, allToneThree(firstTones) {
            drafts[0].tone = 2
        } else if secondTones.count == 2, allToneThree(secondTones) {
            drafts[firstCount].tone = 2
        } else if secondTones.count == 2,
                  !allToneThree(secondTones),
                  secondTones[0] == 3,
                  firstTones.last == 3 {
            drafts[firstCount - 1].tone = 2
        }
    }

    private static func applyErhua(
        to drafts: inout [GPTSoVITSChineseSyllableDraft],
        word: String,
        pos: String,
        lexicon: GPTSoVITSChineseFrontendLexicon?
    ) {
        let characters = Array(word)
        guard drafts.count == characters.count, !drafts.isEmpty else { return }
        let mustErhua = lexicon?.mustErhuaWords ?? Set<String>()
        let notErhua = lexicon?.notErhuaWords ?? Set<String>()

        if characters.last == "儿", drafts.last?.tone == 1 {
            drafts[drafts.count - 1].tone = 2
        }
        if !mustErhua.contains(word) && (notErhua.contains(word) || ["a", "j", "nr"].contains(pos)) {
            return
        }
        guard characters.last == "儿", drafts.count >= 2 else { return }
        let suffix = String(word.suffix(2))
        if [2, 5].contains(drafts[drafts.count - 1].tone), !notErhua.contains(suffix) {
            drafts[drafts.count - 1].tone = drafts[drafts.count - 2].tone
        }
    }

    private static func splitWord(
        _ word: String,
        lexicon: GPTSoVITSChineseFrontendLexicon?
    ) -> [String] {
        let characters = Array(word)
        guard characters.count > 1 else { return [word] }
        guard let lexicon else {
            return [String(characters.prefix(1)), String(characters.dropFirst(1))]
        }

        let segmented = segmentSpanWithLexicon(word, globalStart: 0, lexicon: lexicon).units
        var searchWords = [String]()
        for unit in segmented {
            let tokenCharacters = Array(unit.text)
            if tokenCharacters.count > 2 {
                for start in 0..<(tokenCharacters.count - 1) {
                    let candidate = String(tokenCharacters[start..<(start + 2)])
                    if lexicon.wordFrequency[candidate] != nil {
                        searchWords.append(candidate)
                    }
                }
            }
            if tokenCharacters.count > 3 {
                for start in 0..<(tokenCharacters.count - 2) {
                    let candidate = String(tokenCharacters[start..<(start + 3)])
                    if lexicon.wordFrequency[candidate] != nil {
                        searchWords.append(candidate)
                    }
                }
            }
            searchWords.append(unit.text)
        }

        let sortedSearchWords = searchWords.sorted { lhs, rhs in
            if lhs.count == rhs.count {
                return word.range(of: lhs)?.lowerBound ?? word.startIndex
                    < word.range(of: rhs)?.lowerBound ?? word.startIndex
            }
            return lhs.count < rhs.count
        }
        guard let firstSubword = sortedSearchWords.first,
              let range = word.range(of: firstSubword) else {
            return [String(characters.prefix(1)), String(characters.dropFirst(1))]
        }

        let firstBeginIndex = word.distance(from: word.startIndex, to: range.lowerBound)
        if firstBeginIndex == 0 {
            return [firstSubword, String(word[range.upperBound...])]
        }
        return [String(word[..<range.lowerBound]), firstSubword]
    }

    private static func allToneThree(_ tones: [Int]) -> Bool {
        !tones.isEmpty && tones.allSatisfy { $0 == 3 }
    }

    private static func isReduplication(_ word: String) -> Bool {
        let characters = Array(word)
        return characters.count == 2 && characters[0] == characters[1]
    }

    private static func mapSyllableDraftToPhones(_ draft: GPTSoVITSChineseSyllableDraft) -> [String] {
        if let literalPhone = draft.literalPhone {
            return [literalPhone]
        }
        guard let initialPhone = draft.initialPhone,
              let finalPhoneBase = draft.finalPhoneBase else {
            return ["UNK"]
        }
        return [
            initialPhone,
            finalPhoneBase + String(draft.tone),
        ]
    }

    private static func phoneTemplate(from syllable: GPTSoVITSToneNumberSyllable) -> GPTSoVITSPhoneSyllableTemplate? {
        guard let phones = GPTSoVITSPhoneSymbolAssets.pinyinToPhones[syllable.base], phones.count == 2 else {
            return nil
        }
        return GPTSoVITSPhoneSyllableTemplate(
            label: syllable.base + String(syllable.tone),
            initialPhone: phones[0],
            finalPhoneBase: phones[1],
            tone: syllable.tone
        )
    }

    private static func toneNumberPinyin(for character: Character) -> String? {
        if character == "儿" {
            return "er2"
        }
        let mutable = NSMutableString(string: String(character))
        CFStringTransform(mutable, nil, kCFStringTransformToLatin, false)
        let latin = String(mutable).precomposedStringWithCanonicalMapping.lowercased()

        var base = ""
        var tone = 5
        var hasLetter = false
        for character in latin {
            if let marked = toneMarkedVowels[character] {
                base.append(marked.base)
                tone = marked.tone
                hasLetter = true
                continue
            }
            switch character {
            case "ü":
                base.append("v")
                hasLetter = true
            case "a"..."z":
                base.append(character)
                hasLetter = true
            default:
                continue
            }
        }
        guard hasLetter else { return nil }
        return base + String(tone)
    }

    private static func isPhonePunctuation(
        _ character: Character,
        lexicon: GPTSoVITSChineseFrontendLexicon?
    ) -> Bool {
        phonePunctuationSymbols.contains(character) || lexicon?.punctuation.contains(character) == true
    }

    private static func punctuationCharacters(
        lexicon: GPTSoVITSChineseFrontendLexicon?
    ) -> Set<Character> {
        phonePunctuationSymbols.union(lexicon?.punctuation ?? [])
    }

    private static let phonePunctuationSymbols: Set<Character> = [
        "!", ",", "-", ".", "?", "…",
        "，", "。", "？", "！", "：", "；", "、"
    ]

    private static let jiebaInternalPunctuation: Set<Character> = [
        "."
    ]

    private static let toneMarkedVowels: [Character: (base: Character, tone: Int)] = [
        "ā": ("a", 1), "á": ("a", 2), "ǎ": ("a", 3), "à": ("a", 4),
        "ē": ("e", 1), "é": ("e", 2), "ě": ("e", 3), "è": ("e", 4),
        "ī": ("i", 1), "í": ("i", 2), "ǐ": ("i", 3), "ì": ("i", 4),
        "ō": ("o", 1), "ó": ("o", 2), "ǒ": ("o", 3), "ò": ("o", 4),
        "ū": ("u", 1), "ú": ("u", 2), "ǔ": ("u", 3), "ù": ("u", 4),
        "ǖ": ("v", 1), "ǘ": ("v", 2), "ǚ": ("v", 3), "ǜ": ("v", 4),
        "ê": ("e", 5),
    ]

    private static func isChineseScalarCharacter(_ character: Character) -> Bool {
        character.unicodeScalars.allSatisfy { scalar in
            (0x3400...0x4DBF).contains(Int(scalar.value)) ||
            (0x4E00...0x9FFF).contains(Int(scalar.value))
        }
    }

    private static func isChineseNumericCharacter(_ character: Character) -> Bool {
        chineseNumericCharacters.contains(character)
    }

    private static let chineseNumericCharacters: Set<Character> = [
        "零", "一", "二", "两", "三", "四", "五", "六", "七", "八", "九", "十", "百", "千", "万", "亿"
    ]
}

private struct GPTSoVITSChineseLexicalUnitDraft {
    let unitType: GPTSoVITSPhoneUnitType
    let text: String
    let pos: String?
    let charStart: Int
    let charEnd: Int
    let segmentIndex: Int
}

private struct GPTSoVITSToneNumberSyllable {
    let base: String
    let tone: Int
}

private struct GPTSoVITSChineseSyllableDraft {
    let character: Character
    let unitIndex: Int
    let segmentIndex: Int
    let globalCharIndex: Int
    let literalPhone: String?
    let initialPhone: String?
    let finalPhoneBase: String?
    var tone: Int
}

enum GPTSoVITSPhoneSymbolAssets {
    static let symbols: [String] = symbolsRaw
        .split(separator: "\n", omittingEmptySubsequences: false)
        .map(String.init)

    static let symbolToID: [String: Int] = {
        var mapping = [String: Int]()
        mapping.reserveCapacity(symbols.count)
        for (index, symbol) in symbols.enumerated() {
            mapping[symbol] = index
        }
        return mapping
    }()

    static let pinyinToPhones: [String: [String]] = {
        var mapping = [String: [String]]()
        mapping.reserveCapacity(512)
        for line in opencpopRaw.split(separator: "\n") {
            let items = line.split(separator: "\t")
            guard items.count == 2 else { continue }
            let pinyin = String(items[0])
            let phones = items[1].split(separator: " ").map(String.init)
            mapping[pinyin] = phones
        }
        return mapping
    }()

    private static let symbolsRaw = #"""
!
,
-
.
?
AA
AA0
AA1
AA2
AE0
AE1
AE2
AH0
AH1
AH2
AO0
AO1
AO2
AW0
AW1
AW2
AY0
AY1
AY2
B
CH
D
DH
E1
E2
E3
E4
E5
EE
EH0
EH1
EH2
ER
ER0
ER1
ER2
EY0
EY1
EY2
En1
En2
En3
En4
En5
F
G
HH
I
IH
IH0
IH1
IH2
IY0
IY1
IY2
JH
K
L
M
N
NG
OO
OW0
OW1
OW2
OY0
OY1
OY2
P
R
S
SH
SP
SP2
SP3
T
TH
U
UH0
UH1
UH2
UNK
UW0
UW1
UW2
V
W
Y
Z
ZH
_
a
a1
a2
a3
a4
a5
ai1
ai2
ai3
ai4
ai5
an1
an2
an3
an4
an5
ang1
ang2
ang3
ang4
ang5
ao1
ao2
ao3
ao4
ao5
b
by
c
ch
cl
d
dy
e
e1
e2
e3
e4
e5
ei1
ei2
ei3
ei4
ei5
en1
en2
en3
en4
en5
eng1
eng2
eng3
eng4
eng5
er1
er2
er3
er4
er5
f
g
gy
h
hy
i
i01
i02
i03
i04
i05
i1
i2
i3
i4
i5
ia1
ia2
ia3
ia4
ia5
ian1
ian2
ian3
ian4
ian5
iang1
iang2
iang3
iang4
iang5
iao1
iao2
iao3
iao4
iao5
ie1
ie2
ie3
ie4
ie5
in1
in2
in3
in4
in5
ing1
ing2
ing3
ing4
ing5
iong1
iong2
iong3
iong4
iong5
ir1
ir2
ir3
ir4
ir5
iu1
iu2
iu3
iu4
iu5
j
k
ky
l
m
my
n
ny
o
o1
o2
o3
o4
o5
ong1
ong2
ong3
ong4
ong5
ou1
ou2
ou3
ou4
ou5
p
py
q
r
ry
s
sh
t
ts
u
u1
u2
u3
u4
u5
ua1
ua2
ua3
ua4
ua5
uai1
uai2
uai3
uai4
uai5
uan1
uan2
uan3
uan4
uan5
uang1
uang2
uang3
uang4
uang5
ui1
ui2
ui3
ui4
ui5
un1
un2
un3
un4
un5
uo1
uo2
uo3
uo4
uo5
v
v1
v2
v3
v4
v5
van1
van2
van3
van4
van5
ve1
ve2
ve3
ve4
ve5
vn1
vn2
vn3
vn4
vn5
w
x
y
z
zh
…
[
]
ㄱ
ㄲ
ㄴ
ㄷ
ㄸ
ㄹ
ㅁ
ㅂ
ㅃ
ㅅ
ㅆ
ㅇ
ㅈ
ㅉ
ㅊ
ㅋ
ㅌ
ㅍ
ㅎ
ㅏ
ㅐ
ㅓ
ㅔ
ㅗ
ㅜ
ㅡ
ㅣ
停
空
Ya
Ya1
Ya2
Ya3
Ya4
Ya5
Ya6
Yaa
Yaa1
Yaa2
Yaa3
Yaa4
Yaa5
Yaa6
Yaai1
Yaai2
Yaai3
Yaai4
Yaai5
Yaai6
Yaak1
Yaak2
Yaak3
Yaak4
Yaak5
Yaak6
Yaam1
Yaam2
Yaam3
Yaam4
Yaam5
Yaam6
Yaan1
Yaan2
Yaan3
Yaan4
Yaan5
Yaan6
Yaang1
Yaang2
Yaang3
Yaang4
Yaang5
Yaang6
Yaap1
Yaap2
Yaap3
Yaap4
Yaap5
Yaap6
Yaat1
Yaat2
Yaat3
Yaat4
Yaat5
Yaat6
Yaau1
Yaau2
Yaau3
Yaau4
Yaau5
Yaau6
Yai
Yai1
Yai2
Yai3
Yai4
Yai5
Yai6
Yak
Yak1
Yak2
Yak3
Yak4
Yak5
Yak6
Yam1
Yam2
Yam3
Yam4
Yam5
Yam6
Yan1
Yan2
Yan3
Yan4
Yan5
Yan6
Yang1
Yang2
Yang3
Yang4
Yang5
Yang6
Yap1
Yap2
Yap3
Yap4
Yap5
Yap6
Yat1
Yat2
Yat3
Yat4
Yat5
Yat6
Yau
Yau1
Yau2
Yau3
Yau4
Yau5
Yau6
Yb
Yc
Yd
Ye
Ye1
Ye2
Ye3
Ye4
Ye5
Ye6
Yei1
Yei2
Yei3
Yei4
Yei5
Yei6
Yek1
Yek2
Yek3
Yek4
Yek5
Yek6
Yeng1
Yeng2
Yeng3
Yeng4
Yeng5
Yeng6
Yeoi1
Yeoi2
Yeoi3
Yeoi4
Yeoi5
Yeoi6
Yeon1
Yeon2
Yeon3
Yeon4
Yeon5
Yeon6
Yeot1
Yeot2
Yeot3
Yeot4
Yeot5
Yeot6
Yf
Yg
Yg1
Yg2
Yg3
Yg4
Yg5
Yg6
Ygw
Yh
Yi1
Yi2
Yi3
Yi4
Yi5
Yi6
Yik1
Yik2
Yik3
Yik4
Yik5
Yik6
Yim1
Yim2
Yim3
Yim4
Yim5
Yim6
Yin1
Yin2
Yin3
Yin4
Yin5
Yin6
Ying1
Ying2
Ying3
Ying4
Ying5
Ying6
Yip1
Yip2
Yip3
Yip4
Yip5
Yip6
Yit1
Yit2
Yit3
Yit4
Yit5
Yit6
Yiu1
Yiu2
Yiu3
Yiu4
Yiu5
Yiu6
Yj
Yk
Yk1
Yk2
Yk3
Yk4
Yk5
Yk6
Ykw
Yl
Ym
Ym1
Ym2
Ym3
Ym4
Ym5
Ym6
Yn
Yn1
Yn2
Yn3
Yn4
Yn5
Yn6
Yng
Yo
Yo1
Yo2
Yo3
Yo4
Yo5
Yo6
Yoe1
Yoe2
Yoe3
Yoe4
Yoe5
Yoe6
Yoek1
Yoek2
Yoek3
Yoek4
Yoek5
Yoek6
Yoeng1
Yoeng2
Yoeng3
Yoeng4
Yoeng5
Yoeng6
Yoi
Yoi1
Yoi2
Yoi3
Yoi4
Yoi5
Yoi6
Yok
Yok1
Yok2
Yok3
Yok4
Yok5
Yok6
Yon
Yon1
Yon2
Yon3
Yon4
Yon5
Yon6
Yong1
Yong2
Yong3
Yong4
Yong5
Yong6
Yot1
Yot2
Yot3
Yot4
Yot5
Yot6
You
You1
You2
You3
You4
You5
You6
Yp
Yp1
Yp2
Yp3
Yp4
Yp5
Yp6
Ys
Yt
Yt1
Yt2
Yt3
Yt4
Yt5
Yt6
Yu1
Yu2
Yu3
Yu4
Yu5
Yu6
Yui1
Yui2
Yui3
Yui4
Yui5
Yui6
Yuk
Yuk1
Yuk2
Yuk3
Yuk4
Yuk5
Yuk6
Yun1
Yun2
Yun3
Yun4
Yun5
Yun6
Yung1
Yung2
Yung3
Yung4
Yung5
Yung6
Yut1
Yut2
Yut3
Yut4
Yut5
Yut6
Yw
Yyu1
Yyu2
Yyu3
Yyu4
Yyu5
Yyu6
Yyun1
Yyun2
Yyun3
Yyun4
Yyun5
Yyun6
Yyut1
Yyut2
Yyut3
Yyut4
Yyut5
Yyut6
Yz
"""#

    private static let opencpopRaw = #"""
a	AA a
ai	AA ai
an	AA an
ang	AA ang
ao	AA ao
ba	b a
bai	b ai
ban	b an
bang	b ang
bao	b ao
bei	b ei
ben	b en
beng	b eng
bi	b i
bian	b ian
biao	b iao
bie	b ie
bin	b in
bing	b ing
bo	b o
bu	b u
ca	c a
cai	c ai
can	c an
cang	c ang
cao	c ao
ce	c e
cei	c ei
cen	c en
ceng	c eng
cha	ch a
chai	ch ai
chan	ch an
chang	ch ang
chao	ch ao
che	ch e
chen	ch en
cheng	ch eng
chi	ch ir
chong	ch ong
chou	ch ou
chu	ch u
chua	ch ua
chuai	ch uai
chuan	ch uan
chuang	ch uang
chui	ch ui
chun	ch un
chuo	ch uo
ci	c i0
cong	c ong
cou	c ou
cu	c u
cuan	c uan
cui	c ui
cun	c un
cuo	c uo
da	d a
dai	d ai
dan	d an
dang	d ang
dao	d ao
de	d e
dei	d ei
den	d en
deng	d eng
di	d i
dia	d ia
dian	d ian
diao	d iao
die	d ie
ding	d ing
diu	d iu
dong	d ong
dou	d ou
du	d u
duan	d uan
dui	d ui
dun	d un
duo	d uo
e	EE e
ei	EE ei
en	EE en
eng	EE eng
er	EE er
fa	f a
fan	f an
fang	f ang
fei	f ei
fen	f en
feng	f eng
fo	f o
fou	f ou
fu	f u
ga	g a
gai	g ai
gan	g an
gang	g ang
gao	g ao
ge	g e
gei	g ei
gen	g en
geng	g eng
gong	g ong
gou	g ou
gu	g u
gua	g ua
guai	g uai
guan	g uan
guang	g uang
gui	g ui
gun	g un
guo	g uo
ha	h a
hai	h ai
han	h an
hang	h ang
hao	h ao
he	h e
hei	h ei
hen	h en
heng	h eng
hong	h ong
hou	h ou
hu	h u
hua	h ua
huai	h uai
huan	h uan
huang	h uang
hui	h ui
hun	h un
huo	h uo
ji	j i
jia	j ia
jian	j ian
jiang	j iang
jiao	j iao
jie	j ie
jin	j in
jing	j ing
jiong	j iong
jiu	j iu
ju	j v
jv	j v
juan	j van
jvan	j van
jue	j ve
jve	j ve
jun	j vn
jvn	j vn
ka	k a
kai	k ai
kan	k an
kang	k ang
kao	k ao
ke	k e
kei	k ei
ken	k en
keng	k eng
kong	k ong
kou	k ou
ku	k u
kua	k ua
kuai	k uai
kuan	k uan
kuang	k uang
kui	k ui
kun	k un
kuo	k uo
la	l a
lai	l ai
lan	l an
lang	l ang
lao	l ao
le	l e
lei	l ei
leng	l eng
li	l i
lia	l ia
lian	l ian
liang	l iang
liao	l iao
lie	l ie
lin	l in
ling	l ing
liu	l iu
lo	l o
long	l ong
lou	l ou
lu	l u
luan	l uan
lun	l un
luo	l uo
lv	l v
lve	l ve
ma	m a
mai	m ai
man	m an
mang	m ang
mao	m ao
me	m e
mei	m ei
men	m en
meng	m eng
mi	m i
mian	m ian
miao	m iao
mie	m ie
min	m in
ming	m ing
miu	m iu
mo	m o
mou	m ou
mu	m u
na	n a
nai	n ai
nan	n an
nang	n ang
nao	n ao
ne	n e
nei	n ei
nen	n en
neng	n eng
ni	n i
nian	n ian
niang	n iang
niao	n iao
nie	n ie
nin	n in
ning	n ing
niu	n iu
nong	n ong
nou	n ou
nu	n u
nuan	n uan
nun	n un
nuo	n uo
nv	n v
nve	n ve
o	OO o
ou	OO ou
pa	p a
pai	p ai
pan	p an
pang	p ang
pao	p ao
pei	p ei
pen	p en
peng	p eng
pi	p i
pian	p ian
piao	p iao
pie	p ie
pin	p in
ping	p ing
po	p o
pou	p ou
pu	p u
qi	q i
qia	q ia
qian	q ian
qiang	q iang
qiao	q iao
qie	q ie
qin	q in
qing	q ing
qiong	q iong
qiu	q iu
qu	q v
qv	q v
quan	q van
qvan	q van
que	q ve
qve	q ve
qun	q vn
qvn	q vn
ran	r an
rang	r ang
rao	r ao
re	r e
ren	r en
reng	r eng
ri	r ir
rong	r ong
rou	r ou
ru	r u
rua	r ua
ruan	r uan
rui	r ui
run	r un
ruo	r uo
sa	s a
sai	s ai
san	s an
sang	s ang
sao	s ao
se	s e
sen	s en
seng	s eng
sha	sh a
shai	sh ai
shan	sh an
shang	sh ang
shao	sh ao
she	sh e
shei	sh ei
shen	sh en
sheng	sh eng
shi	sh ir
shou	sh ou
shu	sh u
shua	sh ua
shuai	sh uai
shuan	sh uan
shuang	sh uang
shui	sh ui
shun	sh un
shuo	sh uo
si	s i0
song	s ong
sou	s ou
su	s u
suan	s uan
sui	s ui
sun	s un
suo	s uo
ta	t a
tai	t ai
tan	t an
tang	t ang
tao	t ao
te	t e
tei	t ei
teng	t eng
ti	t i
tian	t ian
tiao	t iao
tie	t ie
ting	t ing
tong	t ong
tou	t ou
tu	t u
tuan	t uan
tui	t ui
tun	t un
tuo	t uo
wa	w a
wai	w ai
wan	w an
wang	w ang
wei	w ei
wen	w en
weng	w eng
wo	w o
wu	w u
xi	x i
xia	x ia
xian	x ian
xiang	x iang
xiao	x iao
xie	x ie
xin	x in
xing	x ing
xiong	x iong
xiu	x iu
xu	x v
xv	x v
xuan	x van
xvan	x van
xue	x ve
xve	x ve
xun	x vn
xvn	x vn
ya	y a
yan	y En
yang	y ang
yao	y ao
ye	y E
yi	y i
yin	y in
ying	y ing
yo	y o
yong	y ong
you	y ou
yu	y v
yv	y v
yuan	y van
yvan	y van
yue	y ve
yve	y ve
yun	y vn
yvn	y vn
za	z a
zai	z ai
zan	z an
zang	z ang
zao	z ao
ze	z e
zei	z ei
zen	z en
zeng	z eng
zha	zh a
zhai	zh ai
zhan	zh an
zhang	zh ang
zhao	zh ao
zhe	zh e
zhei	zh ei
zhen	zh en
zheng	zh eng
zhi	zh ir
zhong	zh ong
zhou	zh ou
zhu	zh u
zhua	zh ua
zhuai	zh uai
zhuan	zh uan
zhuang	zh uang
zhui	zh ui
zhun	zh un
zhuo	zh uo
zi	z i0
zong	z ong
zou	z ou
zu	z u
zuan	z uan
zui	z ui
zun	z un
zuo	z uo
"""#
}
