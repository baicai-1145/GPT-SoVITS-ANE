import Foundation

public enum GPTSoVITSTextLanguage: String {
    case zh
    case allZh = "all_zh"
    case yue
    case allYue = "all_yue"
    case ja
    case allJa = "all_ja"
    case ko
    case allKo = "all_ko"
    case en
    case auto
    case autoYue = "auto_yue"

    public var baseLanguage: String {
        switch self {
        case .allZh:
            return "zh"
        case .allYue:
            return "yue"
        case .allJa:
            return "ja"
        case .allKo:
            return "ko"
        default:
            return rawValue
        }
    }

    public var isChineseLike: Bool {
        switch self {
        case .zh, .allZh, .yue, .allYue:
            return true
        default:
            return false
        }
    }

    public var sentencePrefix: String {
        baseLanguage == "en" ? "." : "。"
    }

    public var sentenceSuffix: String {
        baseLanguage == "en" ? "." : "。"
    }
}

public enum GPTSoVITSTextSplitMethod: String {
    case cut0
    case cut1
    case cut2
    case cut3
    case cut4
    case cut5
}

public struct GPTSoVITSTextSegment {
    public let index: Int
    public let text: String
}

public struct GPTSoVITSTextPreprocessResult {
    public let originalText: String
    public let normalizedInputText: String
    public let splitMethod: GPTSoVITSTextSplitMethod
    public let language: GPTSoVITSTextLanguage
    public let segments: [GPTSoVITSTextSegment]
}

public final class GPTSoVITSTextFrontend {
    public let maxSegmentLength: Int
    public let minMergedSegmentLength: Int

    public init(
        maxSegmentLength: Int = 510,
        minMergedSegmentLength: Int = 5
    ) {
        self.maxSegmentLength = maxSegmentLength
        self.minMergedSegmentLength = minMergedSegmentLength
    }

    public func preprocess(
        text: String,
        language: GPTSoVITSTextLanguage,
        splitMethod: GPTSoVITSTextSplitMethod
    ) -> GPTSoVITSTextPreprocessResult {
        let normalizedInputText = preprocessInputText(text, language: language)
        let segments = preSegment(
            text: normalizedInputText,
            language: language,
            splitMethod: splitMethod
        ).enumerated().map { index, text in
            GPTSoVITSTextSegment(index: index, text: text)
        }
        return GPTSoVITSTextPreprocessResult(
            originalText: text,
            normalizedInputText: normalizedInputText,
            splitMethod: splitMethod,
            language: language,
            segments: segments
        )
    }

    public func normalizeInputText(
        _ text: String,
        language: GPTSoVITSTextLanguage,
        traditionalToSimplifiedMap: [String: String]? = nil
    ) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }
        if language.isChineseLike {
            return normalizeChineseLikeText(
                trimmed,
                traditionalToSimplifiedMap: traditionalToSimplifiedMap
            )
        }
        return trimmed.replacingOccurrences(
            of: #"[ ]{2,}"#,
            with: " ",
            options: .regularExpression
        )
    }

    public func preprocessInputText(
        _ text: String,
        language: GPTSoVITSTextLanguage
    ) -> String {
        let trimmedNewlines = text.trimmingCharacters(in: CharacterSet(charactersIn: "\n"))
        guard !trimmedNewlines.isEmpty else { return "" }
        if language.isChineseLike {
            return replaceConsecutivePunctuation(trimmedNewlines)
        }
        return trimmedNewlines
    }

    public func preSegment(
        text: String,
        language: GPTSoVITSTextLanguage,
        splitMethod: GPTSoVITSTextSplitMethod
    ) -> [String] {
        var text = text.trimmingCharacters(in: CharacterSet(charactersIn: "\n"))
        guard !text.isEmpty else { return [] }

        if !Self.splitCharacters.contains(text.first ?? Character(" ")),
           firstClause(in: text).count < 4 {
            text = language.sentencePrefix + text
        }

        text = applySplitMethod(splitMethod, to: text)
        while text.contains("\n\n") {
            text = text.replacingOccurrences(of: "\n\n", with: "\n")
        }

        let segments = mergeShortTexts(
            filterText(text.split(separator: "\n").map(String.init)),
            threshold: minMergedSegmentLength
        )

        var finalized = [String]()
        finalized.reserveCapacity(segments.count)
        for segment in segments {
            let candidate = segment.trimmingCharacters(in: CharacterSet(charactersIn: "\n"))
            guard !candidate.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }
            guard candidate.range(of: #"\W+"#, options: .regularExpression) == nil || candidate.contains(where: { $0.isLetter || $0.isNumber }) else {
                continue
            }
            var completed = candidate
            if !Self.splitCharacters.contains(completed.last ?? Character(" ")) {
                completed += language.sentenceSuffix
            }
            if completed.count > maxSegmentLength {
                finalized.append(contentsOf: splitBigText(completed, maxLength: maxSegmentLength))
            } else {
                finalized.append(completed)
            }
        }
        return finalized
    }

    public func replaceConsecutivePunctuation(_ text: String) -> String {
        var result = ""
        var previousWasPunctuation = false
        for character in text {
            let currentIsPunctuation = Self.punctuationCharacters.contains(character)
            if currentIsPunctuation && previousWasPunctuation {
                continue
            }
            result.append(character)
            previousWasPunctuation = currentIsPunctuation
        }
        return result
    }

    private func normalizeChineseLikeText(
        _ text: String,
        traditionalToSimplifiedMap: [String: String]?
    ) -> String {
        let replaced = text
            .replacingOccurrences(of: "嗯", with: "恩")
            .replacingOccurrences(of: "呣", with: "母")
        let punctuationMapped = Self.chineseReplacementMap.reduce(replaced) { partial, pair in
            partial.replacingOccurrences(of: pair.key, with: pair.value)
        }
        let digitMapped = punctuationMapped.map { character in
            Self.chineseDigitMap[character] ?? String(character)
        }.joined()
        let simplified = applyTraditionalToSimplifiedMap(
            digitMapped,
            mapping: traditionalToSimplifiedMap
        )
        return String(
            simplified.filter { character in
                Self.isChineseCharacter(character) || Self.punctuationCharacters.contains(character)
            }
        )
    }

    private func applyTraditionalToSimplifiedMap(
        _ text: String,
        mapping: [String: String]?
    ) -> String {
        guard let mapping, !mapping.isEmpty else {
            return text
        }
        var converted = String()
        converted.reserveCapacity(text.count)
        for character in text {
            converted += mapping[String(character)] ?? String(character)
        }
        return converted
    }

    private func applySplitMethod(_ method: GPTSoVITSTextSplitMethod, to input: String) -> String {
        switch method {
        case .cut0:
            return input
        case .cut1:
            return splitEveryFourSentences(input)
        case .cut2:
            return splitEveryFiftyCharacters(input)
        case .cut3:
            return input
                .trimmingCharacters(in: CharacterSet(charactersIn: "\n"))
                .trimmingCharacters(in: CharacterSet(charactersIn: "。"))
                .split(separator: "。")
                .map(String.init)
                .filter { !isOnlyPunctuation($0) }
                .joined(separator: "\n")
        case .cut4:
            let trimmed = input
                .trimmingCharacters(in: CharacterSet(charactersIn: "\n"))
                .trimmingCharacters(in: CharacterSet(charactersIn: "."))
            let range = NSRange(trimmed.startIndex..., in: trimmed)
            let regex = try? NSRegularExpression(pattern: #"(?<!\d)\.(?!\d)"#)
            let matches = regex?.matches(in: trimmed, options: [], range: range) ?? []
            if matches.isEmpty {
                return trimmed
            }
            var parts = [String]()
            var lastLocation = trimmed.startIndex
            for match in matches {
                guard let range = Range(match.range, in: trimmed) else { continue }
                parts.append(String(trimmed[lastLocation..<range.lowerBound]))
                lastLocation = range.upperBound
            }
            parts.append(String(trimmed[lastLocation...]))
            return parts.filter { !isOnlyPunctuation($0) }.joined(separator: "\n")
        case .cut5:
            return splitByPunctuation(input)
        }
    }

    private func splitBigText(_ text: String, maxLength: Int) -> [String] {
        let segments = splitKeepingPunctuation(text)
        var result = [String]()
        var current = ""
        for segment in segments {
            if (current + segment).count > maxLength {
                if !current.isEmpty {
                    result.append(current)
                }
                current = segment
            } else {
                current += segment
            }
        }
        if !current.isEmpty {
            result.append(current)
        }
        return result
    }

    private func splitKeepingPunctuation(_ text: String) -> [String] {
        guard !text.isEmpty else { return [] }
        let replaced = text.replacingOccurrences(of: "……", with: "。").replacingOccurrences(of: "——", with: "，")
        var working = replaced
        if let last = working.last, !Self.splitCharacters.contains(last) {
            working += "。"
        }
        var result = [String]()
        var start = working.startIndex
        var cursor = working.startIndex
        while cursor < working.endIndex {
            let next = working.index(after: cursor)
            if Self.splitCharacters.contains(working[cursor]) {
                result.append(String(working[start..<next]))
                start = next
            }
            cursor = next
        }
        return result
    }

    private func splitEveryFourSentences(_ text: String) -> String {
        let parts = splitKeepingPunctuation(text)
        guard parts.count >= 2 else { return text }
        var groups = [String]()
        var index = 0
        while index < parts.count {
            let end = min(index + 4, parts.count)
            groups.append(parts[index..<end].joined())
            index = end
        }
        return groups.filter { !isOnlyPunctuation($0) }.joined(separator: "\n")
    }

    private func splitEveryFiftyCharacters(_ text: String) -> String {
        let parts = splitKeepingPunctuation(text)
        guard parts.count >= 2 else { return text }
        var result = [String]()
        var current = ""
        var length = 0
        for part in parts {
            current += part
            length += part.count
            if length > 50 {
                result.append(current)
                current = ""
                length = 0
            }
        }
        if !current.isEmpty {
            result.append(current)
        }
        if result.count > 1, let last = result.last, last.count < 50 {
            result[result.count - 2] += last
            result.removeLast()
        }
        return result.filter { !isOnlyPunctuation($0) }.joined(separator: "\n")
    }

    private func splitByPunctuation(_ text: String) -> String {
        let punctuation = Self.pundCharacters
        var items = [String]()
        var current = ""
        let characters = Array(text)
        for index in characters.indices {
            let character = characters[index]
            if punctuation.contains(character) {
                if character == ".", index > characters.startIndex, index < characters.index(before: characters.endIndex),
                   characters[characters.index(before: index)].isNumber,
                   characters[characters.index(after: index)].isNumber {
                    current.append(character)
                } else {
                    current.append(character)
                    items.append(current)
                    current = ""
                }
            } else {
                current.append(character)
            }
        }
        if !current.isEmpty {
            items.append(current)
        }
        return items.filter { !isOnlyPunctuation($0) }.joined(separator: "\n")
    }

    private func firstClause(in text: String) -> String {
        let parts = text.split(whereSeparator: { Self.splitCharacters.contains($0) })
        return parts.first.map(String.init)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }

    private func filterText(_ texts: [String]) -> [String] {
        texts.compactMap { text in
            text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : text
        }
    }

    private func mergeShortTexts(_ texts: [String], threshold: Int) -> [String] {
        guard texts.count >= 2 else { return texts }
        var result = [String]()
        var current = ""
        for text in texts {
            current += text
            if current.count >= threshold {
                result.append(current)
                current = ""
            }
        }
        if !current.isEmpty {
            if result.isEmpty {
                result.append(current)
            } else {
                result[result.count - 1] += current
            }
        }
        return result
    }

    private func isOnlyPunctuation(_ text: String) -> Bool {
        !text.isEmpty && text.allSatisfy { Self.pundCharacters.contains($0) || $0.isWhitespace }
    }

    private static func isChineseCharacter(_ character: Character) -> Bool {
        character.unicodeScalars.allSatisfy { scalar in
            (0x3400...0x4DBF).contains(Int(scalar.value)) ||
            (0x4E00...0x9FFF).contains(Int(scalar.value))
        }
    }

    private static let splitCharacters: Set<Character> = ["，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"]
    private static let punctuationCharacters: Set<Character> = ["!", "?", "…", ",", ".", "-"]
    private static let pundCharacters: Set<Character> = [",", ".", ";", "?", "!", "、", "，", "。", "？", "！", "：", "…"]
    private static let chineseReplacementMap: [String: String] = [
        "：": ",",
        "；": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "·": ",",
        "、": ",",
        "...": "…",
        "$": ".",
        "/": ",",
        "—": "-",
        "~": "…",
        "～": "…",
    ]
    private static let chineseDigitMap: [Character: String] = [
        "0": "零",
        "1": "一",
        "2": "二",
        "3": "三",
        "4": "四",
        "5": "五",
        "6": "六",
        "7": "七",
        "8": "八",
        "9": "九",
    ]
}
