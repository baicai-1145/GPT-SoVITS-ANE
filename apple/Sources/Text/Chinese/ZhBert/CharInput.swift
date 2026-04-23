import Foundation

public struct GPTSoVITSZhBertCharInput {
    public let normalizedText: String
    public let tokens: [String]
    public let inputIDs: [Int64]
    public let attentionMask: [Int64]
    public let tokenTypeIDs: [Int64]
    public let padTokenID: Int64

    public var textCharacterCount: Int {
        normalizedText.count
    }

    public var tokenCount: Int {
        inputIDs.count
    }
}

public struct GPTSoVITSWordPieceTokenizationResult {
    public let tokens: [String]
    public let textToToken: [Int?]
    public let tokenToText: [Range<Int>]
}

public struct GPTSoVITSChineseZhBertCharSegmentResult {
    public let segment: GPTSoVITSTextSegment
    public let phoneResult: GPTSoVITSChinesePhoneResult
    public let bertInput: GPTSoVITSZhBertCharInput
}

public struct GPTSoVITSChineseZhBertCharPreprocessResult {
    public let preprocessResult: GPTSoVITSTextPreprocessResult
    public let segmentResults: [GPTSoVITSChineseZhBertCharSegmentResult]
}

public enum GPTSoVITSZhBertTokenizerError: LocalizedError {
    case invalidTokenizerFormat
    case missingSpecialToken(String)

    public var errorDescription: String? {
        switch self {
        case .invalidTokenizerFormat:
            return "tokenizer.json 不是当前支持的 WordPiece 词表格式。"
        case let .missingSpecialToken(token):
            return "tokenizer.json 缺少特殊 token \(token)。"
        }
    }
}

public final class GPTSoVITSZhBertTokenizer {
    public let tokenizerJSONURL: URL

    private let vocabulary: [String: Int]
    private let padTokenID: Int
    private let clsTokenID: Int
    private let sepTokenID: Int
    private let unkTokenID: Int

    public init(tokenizerJSONURL: URL) throws {
        self.tokenizerJSONURL = tokenizerJSONURL
        let data = try GPTSoVITSRuntimeProfiler.measure("zhbert_tokenizer.read") {
            try Data(contentsOf: tokenizerJSONURL)
        }
        let decoder = JSONDecoder()
        let config = try GPTSoVITSRuntimeProfiler.measure("zhbert_tokenizer.decode") {
            try decoder.decode(GPTSoVITSZhBertTokenizerConfig.self, from: data)
        }
        guard config.model.type == "WordPiece" else {
            throw GPTSoVITSZhBertTokenizerError.invalidTokenizerFormat
        }

        let resolvedVocabulary = config.model.vocab
        vocabulary = resolvedVocabulary
        guard let padTokenID = resolvedVocabulary["[PAD]"] else {
            throw GPTSoVITSZhBertTokenizerError.missingSpecialToken("[PAD]")
        }
        guard let clsTokenID = resolvedVocabulary["[CLS]"] else {
            throw GPTSoVITSZhBertTokenizerError.missingSpecialToken("[CLS]")
        }
        guard let sepTokenID = resolvedVocabulary["[SEP]"] else {
            throw GPTSoVITSZhBertTokenizerError.missingSpecialToken("[SEP]")
        }
        guard let unkTokenID = resolvedVocabulary[config.model.unkToken] ?? resolvedVocabulary["[UNK]"] else {
            throw GPTSoVITSZhBertTokenizerError.missingSpecialToken("[UNK]")
        }

        self.padTokenID = padTokenID
        self.clsTokenID = clsTokenID
        self.sepTokenID = sepTokenID
        self.unkTokenID = unkTokenID
    }

    public convenience init(modelDirectory: URL) throws {
        try self.init(tokenizerJSONURL: modelDirectory.appendingPathComponent("tokenizer.json"))
    }

    public func tokenID(for token: String) -> Int {
        vocabulary[token] ?? unkTokenID
    }

    public func convertTokensToIDs(_ tokens: [String]) -> [Int64] {
        tokens.map { Int64(tokenID(for: $0)) }
    }

    public func tokenizeWordPiece(_ word: String) -> [String] {
        let lowered = word.lowercased()
        guard !lowered.isEmpty else { return [] }
        if vocabulary[lowered] != nil {
            return [lowered]
        }

        let characters = Array(lowered)
        var tokens = [String]()
        var start = 0
        while start < characters.count {
            var end = characters.count
            var currentToken: String?
            while end > start {
                let piece = String(characters[start..<end])
                let candidate = start == 0 ? piece : "##" + piece
                if vocabulary[candidate] != nil {
                    currentToken = candidate
                    break
                }
                end -= 1
            }
            guard let currentToken else {
                return ["[UNK]"]
            }
            tokens.append(currentToken)
            start = end
        }
        return tokens
    }

    public func tokenizeAndMap(text: String) -> GPTSoVITSWordPieceTokenizationResult {
        let wordization = wordizeAndMap(text: text.lowercased())
        var tokens = [String]()
        var tokenToText = [Range<Int>]()

        for (word, wordRange) in zip(wordization.words, wordization.wordToText) {
            let wordTokens = tokenizeWordPiece(word)
            if wordTokens.isEmpty || wordTokens == ["[UNK]"] {
                tokens.append("[UNK]")
                tokenToText.append(wordRange)
                continue
            }

            var currentWordStart = wordRange.lowerBound
            for wordToken in wordTokens {
                let tokenBody = wordToken.hasPrefix("##") ? String(wordToken.dropFirst(2)) : wordToken
                let tokenLength = tokenBody.count
                let range = currentWordStart..<(currentWordStart + tokenLength)
                tokenToText.append(range)
                tokens.append(wordToken)
                currentWordStart += tokenLength
            }
        }

        var textToToken = wordization.textToWord
        for (tokenIndex, range) in tokenToText.enumerated() {
            for textIndex in range where textIndex < textToToken.count {
                textToToken[textIndex] = tokenIndex
            }
        }

        return GPTSoVITSWordPieceTokenizationResult(
            tokens: tokens,
            textToToken: textToToken,
            tokenToText: tokenToText
        )
    }

    public func prepareInput(normalizedText: String) -> GPTSoVITSZhBertCharInput {
        let textTokens = Array(normalizedText).map { character -> String in
            let token = String(character)
            return vocabulary[token] == nil ? "[UNK]" : token
        }
        let tokens = ["[CLS]"] + textTokens + ["[SEP]"]
        let inputIDs = tokens.map { token -> Int64 in
            switch token {
            case "[CLS]":
                return Int64(clsTokenID)
            case "[SEP]":
                return Int64(sepTokenID)
            default:
                return Int64(tokenID(for: token))
            }
        }
        return GPTSoVITSZhBertCharInput(
            normalizedText: normalizedText,
            tokens: tokens,
            inputIDs: inputIDs,
            attentionMask: Array(repeating: 1, count: tokens.count),
            tokenTypeIDs: Array(repeating: 0, count: tokens.count),
            padTokenID: Int64(padTokenID)
        )
    }

    public func prepareInput(phoneResult: GPTSoVITSChinesePhoneResult) -> GPTSoVITSZhBertCharInput {
        prepareInput(normalizedText: phoneResult.normalizedText)
    }
}

public extension GPTSoVITSChinesePhoneResult {
    func prepareZhBertCharInput(
        using tokenizer: GPTSoVITSZhBertTokenizer
    ) -> GPTSoVITSZhBertCharInput {
        tokenizer.prepareInput(phoneResult: self)
    }
}

public extension GPTSoVITSTextFrontend {
    func preprocessChineseZhBertCharSegments(
        text: String,
        language: GPTSoVITSTextLanguage = .zh,
        splitMethod: GPTSoVITSTextSplitMethod = .cut5,
        tokenizer: GPTSoVITSZhBertTokenizer,
        g2pwDriver: (any GPTSoVITSG2PWPredicting)? = nil
    ) throws -> GPTSoVITSChineseZhBertCharPreprocessResult {
        let phoneResult = try preprocessChinesePhoneSegments(
            text: text,
            language: language,
            splitMethod: splitMethod,
            g2pwDriver: g2pwDriver
        )
        let segmentResults = phoneResult.segmentResults.map { item in
            GPTSoVITSChineseZhBertCharSegmentResult(
                segment: item.segment,
                phoneResult: item.phoneResult,
                bertInput: tokenizer.prepareInput(phoneResult: item.phoneResult)
            )
        }
        return GPTSoVITSChineseZhBertCharPreprocessResult(
            preprocessResult: phoneResult.preprocessResult,
            segmentResults: segmentResults
        )
    }
}

private struct GPTSoVITSWordizationResult {
    let words: [String]
    let textToWord: [Int?]
    let wordToText: [Range<Int>]
}

private func wordizeAndMap(text: String) -> GPTSoVITSWordizationResult {
    let characters = Array(text)
    var words = [String]()
    var textToWord = [Int?]()
    var wordToText = [Range<Int>]()
    var cursor = 0

    while cursor < characters.count {
        let character = characters[cursor]
        if character.isWhitespace {
            textToWord.append(nil)
            cursor += 1
            continue
        }

        if character.isASCIIAlphaNumeric {
            let start = cursor
            while cursor < characters.count, characters[cursor].isASCIIAlphaNumeric {
                cursor += 1
            }
            words.append(String(characters[start..<cursor]))
            wordToText.append(start..<cursor)
            textToWord.append(contentsOf: Array(repeating: words.count - 1, count: cursor - start))
            continue
        }

        let next = cursor + 1
        words.append(String(character))
        wordToText.append(cursor..<next)
        textToWord.append(words.count - 1)
        cursor = next
    }

    return GPTSoVITSWordizationResult(
        words: words,
        textToWord: textToWord,
        wordToText: wordToText
    )
}

private extension Character {
    var isASCIIAlphaNumeric: Bool {
        guard unicodeScalars.count == 1, let scalar = unicodeScalars.first, scalar.isASCII else {
            return false
        }
        return CharacterSet.alphanumerics.contains(scalar)
    }
}

private struct GPTSoVITSZhBertTokenizerConfig: Decodable {
    struct Model: Decodable {
        let type: String
        let unkToken: String
        let vocab: [String: Int]

        private enum CodingKeys: String, CodingKey {
            case type
            case unkToken = "unk_token"
            case vocab
        }
    }

    let model: Model
}
