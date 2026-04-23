import CoreML
import Foundation

public struct G2PWBundleManifest: Decodable {
    public struct Artifact: Decodable {
        public let target: String
        public let filename: String
        public let path: String
    }

    public struct Artifacts: Decodable {
        public let model: Artifact
        public let tokenizer: Artifact
        public let runtimeAssets: Artifact

        private enum CodingKeys: String, CodingKey {
            case model
            case tokenizer
            case runtimeAssets = "runtime_assets"
        }
    }

    public struct Runtime: Decodable {
        public struct Shapes: Decodable {
            public let batchSize: Int
            public let tokenLen: Int
            public let labelCount: Int

            private enum CodingKeys: String, CodingKey {
                case batchSize = "batch_size"
                case tokenLen = "token_len"
                case labelCount = "label_count"
            }
        }

        public let shapes: Shapes
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

public struct G2PWRuntimeAssets: Decodable {
    public struct PhoneTemplate: Decodable {
        public let label: String
        public let initialPhone: String
        public let finalPhoneBase: String
        public let tone: Int

        private enum CodingKeys: String, CodingKey {
            case label
            case initialPhone = "initial_phone"
            case finalPhoneBase = "final_phone_base"
            case tone
        }

        public func toFrontendTemplate() -> GPTSoVITSPhoneSyllableTemplate {
            GPTSoVITSPhoneSyllableTemplate(
                label: label,
                initialPhone: initialPhone,
                finalPhoneBase: finalPhoneBase,
                tone: tone
            )
        }
    }

    public struct FrontendLexicon: Decodable {
        public struct Segmenter: Decodable {
            public struct POSSegHMM: Decodable {
                public let charStateTab: [String: [String]]
                public let startProb: [String: Double]
                public let transProb: [String: [String: Double]]
                public let emitProb: [String: [String: Double]]

                private enum CodingKeys: String, CodingKey {
                    case charStateTab = "char_state_tab"
                    case startProb = "start_prob"
                    case transProb = "trans_prob"
                    case emitProb = "emit_prob"
                }

                public func toRuntimeHMM() -> GPTSoVITSPossegHMM {
                    GPTSoVITSPossegHMM(
                        charStateTab: charStateTab,
                        startProb: startProb,
                        transProb: transProb,
                        emitProb: emitProb
                    )
                }
            }

            public let type: String
            public let wordFrequency: [String: Int]
            public let wordPOS: [String: String]
            public let forcedWords: [String]
            public let possegHMM: POSSegHMM?
            public let totalFrequency: Int
            public let maxWordLength: Int

            private enum CodingKeys: String, CodingKey {
                case type
                case wordFrequency = "word_frequency"
                case wordPOS = "word_pos"
                case forcedWords = "forced_words"
                case possegHMM = "posseg_hmm"
                case totalFrequency = "total_frequency"
                case maxWordLength = "max_word_length"
            }
        }

        public struct ToneSandhi: Decodable {
            public let mustNeutralToneWords: [String]
            public let mustNotNeutralToneWords: [String]
            public let punctuation: [String]

            private enum CodingKeys: String, CodingKey {
                case mustNeutralToneWords = "must_neutral_tone_words"
                case mustNotNeutralToneWords = "must_not_neutral_tone_words"
                case punctuation
            }
        }

        public struct Erhua: Decodable {
            public let mustErhuaWords: [String]
            public let notErhuaWords: [String]

            private enum CodingKeys: String, CodingKey {
                case mustErhuaWords = "must_erhua_words"
                case notErhuaWords = "not_erhua_words"
            }
        }

        public struct PhraseUnitBreakdown: Decodable {
            public let text: String
            public let pos: String
            public let charStart: Int
            public let charEnd: Int

            private enum CodingKeys: String, CodingKey {
                case text
                case pos
                case charStart = "char_start"
                case charEnd = "char_end"
            }

            public func toRuntimeBreakdown() -> GPTSoVITSChineseFrontendLexicon.PhraseUnitBreakdown {
                GPTSoVITSChineseFrontendLexicon.PhraseUnitBreakdown(
                    text: text,
                    pos: pos,
                    charStart: charStart,
                    charEnd: charEnd
                )
            }
        }

        public let formatVersion: Int
        public let segmenter: Segmenter
        public let phrasePhoneTemplates: [String: [PhoneTemplate]]
        public let phraseUnitBreakdowns: [String: [PhraseUnitBreakdown]]
        public let toneSandhi: ToneSandhi
        public let erhua: Erhua
        public let traditionalToSimplifiedMap: [String: String]

        private enum CodingKeys: String, CodingKey {
            case formatVersion = "format_version"
            case segmenter
            case phrasePhoneTemplates = "phrase_phone_templates"
            case phraseUnitBreakdowns = "phrase_unit_breakdowns"
            case toneSandhi = "tone_sandhi"
            case erhua
            case traditionalToSimplifiedMap = "traditional_to_simplified_map"
        }

        public func toFrontendLexicon() -> GPTSoVITSChineseFrontendLexicon {
            GPTSoVITSChineseFrontendLexicon(
                wordFrequency: segmenter.wordFrequency,
                wordPOS: segmenter.wordPOS,
                forcedWords: Set(segmenter.forcedWords),
                possegHMM: segmenter.possegHMM?.toRuntimeHMM(),
                totalFrequency: max(Double(segmenter.totalFrequency), 1.0),
                maxWordLength: max(segmenter.maxWordLength, 1),
                phraseTemplates: phrasePhoneTemplates.mapValues { templates in
                    templates.map { $0.toFrontendTemplate() }
                },
                phraseUnitBreakdowns: phraseUnitBreakdowns.mapValues { units in
                    units.map { $0.toRuntimeBreakdown() }
                },
                mustNeutralToneWords: Set(toneSandhi.mustNeutralToneWords),
                mustNotNeutralToneWords: Set(toneSandhi.mustNotNeutralToneWords),
                mustErhuaWords: Set(erhua.mustErhuaWords),
                notErhuaWords: Set(erhua.notErhuaWords),
                punctuation: Set(toneSandhi.punctuation.compactMap(\.first)),
                traditionalToSimplifiedMap: traditionalToSimplifiedMap
            )
        }
    }

    public let formatVersion: Int
    public let style: String
    public let chars: [String]
    public let labels: [String]
    public let labelPhoneTemplates: [PhoneTemplate?]
    public let phonemeIndicesByCharID: [[Int]]
    public let polyphonicChars: [String]
    public let monophonicBopomofo: [String: String]
    public let monophonicPhoneTemplates: [String: PhoneTemplate]
    public let charLabelAliases: [String: [String: String]]
    public let unresolvedLabels: [String]
    public let unresolvedMonophonicBopomofo: [String: String]
    public let polyphonicContextChars: Int
    public let frontendLexicon: FrontendLexicon?

    private enum CodingKeys: String, CodingKey {
        case formatVersion = "format_version"
        case style
        case chars
        case labels
        case labelPhoneTemplates = "label_phone_templates"
        case phonemeIndicesByCharID = "phoneme_indices_by_char_id"
        case polyphonicChars = "polyphonic_chars"
        case monophonicBopomofo = "monophonic_bopomofo"
        case monophonicPhoneTemplates = "monophonic_phone_templates"
        case charLabelAliases = "char_label_aliases"
        case unresolvedLabels = "unresolved_labels"
        case unresolvedMonophonicBopomofo = "unresolved_monophonic_bopomofo"
        case polyphonicContextChars = "polyphonic_context_chars"
        case frontendLexicon = "frontend_lexicon"
    }
}

public final class G2PWCoreMLDriver: GPTSoVITSG2PWPredicting {
    private struct PreparedQueryRow {
        let inputIDs: [Int32]
        let tokenTypeIDs: [Int32]
        let attentionMask: [Int32]
        let phonemeMask: [Float32]
        let charID: Int32
        let positionID: Int32
    }

    private struct TruncatedQueryContext {
        let text: String
        let queryID: Int
        let tokens: [String]
        let textToToken: [Int?]
    }

    public let manifest: G2PWBundleManifest
    public let model: MLModel
    public let tokenizer: GPTSoVITSZhBertTokenizer
    public let runtimeAssets: G2PWRuntimeAssets
    public let frontendLexicon: GPTSoVITSChineseFrontendLexicon?

    private let charToID: [String: Int]
    private let polyphonicCharSet: Set<String>
    private let labelPhoneTemplates: [GPTSoVITSPhoneSyllableTemplate?]
    private let labelTemplateByLabel: [String: GPTSoVITSPhoneSyllableTemplate]
    private let monophonicPhoneTemplates: [String: GPTSoVITSPhoneSyllableTemplate]

    public static func bundleType(at bundleDirectory: URL) -> String? {
        let manifestURL = bundleDirectory.appendingPathComponent("manifest.json")
        guard let data = try? Data(contentsOf: manifestURL) else {
            return nil
        }
        return (try? JSONDecoder().decode(G2PWBundleManifest.self, from: data))?.bundleType
    }

    public static func isBundleDirectory(_ bundleDirectory: URL) -> Bool {
        guard let bundleType = bundleType(at: bundleDirectory) else {
            return false
        }
        return bundleType == "g2pw_bundle" || bundleType == "gpt_sovits_g2pw_coreml_bundle"
    }

    public init(bundleDirectory: URL, configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        let manifestURL = bundleDirectory.appendingPathComponent("manifest.json")
        let manifestData = try GPTSoVITSRuntimeProfiler.measure("g2pw.init.manifest.read") {
            try Data(contentsOf: manifestURL)
        }
        self.manifest = try GPTSoVITSRuntimeProfiler.measure("g2pw.init.manifest.decode") {
            try JSONDecoder().decode(G2PWBundleManifest.self, from: manifestData)
        }

        let modelURL = bundleDirectory.appendingPathComponent(manifest.artifacts.model.filename)
        let tokenizerURL = bundleDirectory.appendingPathComponent(manifest.artifacts.tokenizer.filename)
        let runtimeAssetsURL = bundleDirectory.appendingPathComponent(manifest.artifacts.runtimeAssets.filename)

        self.model = try GPTSoVITSRuntimeProfiler.measure("g2pw.init.model.load") {
            try Self.loadModel(at: modelURL, configuration: configuration)
        }
        self.tokenizer = try GPTSoVITSRuntimeProfiler.measure("g2pw.init.tokenizer.init") {
            try GPTSoVITSZhBertTokenizer(tokenizerJSONURL: tokenizerURL)
        }
        let runtimeAssetsData = try GPTSoVITSRuntimeProfiler.measure("g2pw.init.runtime_assets.read") {
            try Data(contentsOf: runtimeAssetsURL)
        }
        self.runtimeAssets = try GPTSoVITSRuntimeProfiler.measure("g2pw.init.runtime_assets.decode") {
            try JSONDecoder().decode(
                G2PWRuntimeAssets.self,
                from: runtimeAssetsData
            )
        }
        guard manifest.runtime.shapes.labelCount == runtimeAssets.labels.count else {
            throw NSError(domain: "G2PWCoreMLDriver", code: 100, userInfo: [
                NSLocalizedDescriptionKey: "g2pw manifest label_count=\(manifest.runtime.shapes.labelCount) does not match runtime asset labels=\(runtimeAssets.labels.count)."
            ])
        }
        guard runtimeAssets.labels.count == runtimeAssets.labelPhoneTemplates.count else {
            throw NSError(domain: "G2PWCoreMLDriver", code: 101, userInfo: [
                NSLocalizedDescriptionKey: "g2pw runtime asset labels count does not match label_phone_templates count."
            ])
        }
        let derivedAssets = Self.buildDerivedAssets(from: runtimeAssets)
        self.charToID = derivedAssets.charToID
        self.polyphonicCharSet = derivedAssets.polyphonicCharSet
        let labelPhoneTemplates = derivedAssets.labelPhoneTemplates
        self.labelPhoneTemplates = labelPhoneTemplates
        self.labelTemplateByLabel = Self.buildLabelTemplateByLabel(
            labels: runtimeAssets.labels,
            labelPhoneTemplates: labelPhoneTemplates
        )
        self.monophonicPhoneTemplates = derivedAssets.monophonicPhoneTemplates
        self.frontendLexicon = derivedAssets.frontendLexicon
    }

    public func predictSyllableAlignment(for normalizedText: String) throws -> G2PWPredictionResult {
        let characters = Array(normalizedText)
        var bopomofoByChar = Array<String?>(repeating: nil, count: characters.count)
        var phoneTemplateByChar = Array<GPTSoVITSPhoneSyllableTemplate?>(repeating: nil, count: characters.count)
        var polyphonicIndices = [Int]()

        for (index, character) in characters.enumerated() {
            let key = String(character)
            if polyphonicCharSet.contains(key) {
                polyphonicIndices.append(index)
            } else if let monophonic = runtimeAssets.monophonicBopomofo[key] {
                bopomofoByChar[index] = monophonic
                phoneTemplateByChar[index] = monophonicPhoneTemplates[key]
            }
        }

        guard !polyphonicIndices.isEmpty else {
            return G2PWPredictionResult(
                normalizedText: normalizedText,
                bopomofoByChar: bopomofoByChar,
                phoneTemplateByChar: phoneTemplateByChar,
                queryCount: 0,
                g2pwResolvedCount: 0
            )
        }

        let context = buildPolyphonicContext(text: normalizedText, polyphonicIndices: polyphonicIndices)
        let queryRows = try polyphonicIndices.map { globalIndex in
            try buildQueryRow(text: context.text, queryID: globalIndex - context.offset)
        }

        var predictions = [Int]()
        predictions.reserveCapacity(queryRows.count)
        var queryCursor = 0
        while queryCursor < queryRows.count {
            let batchEnd = min(queryCursor + manifest.runtime.shapes.batchSize, queryRows.count)
            predictions.append(contentsOf: try predictBatch(rows: Array(queryRows[queryCursor..<batchEnd])))
            queryCursor = batchEnd
        }

        var resolvedCount = 0
        for (globalIndex, labelIndex) in zip(polyphonicIndices, predictions) {
            let character = String(characters[globalIndex])
            let rawLabel = runtimeAssets.labels[labelIndex]
            let canonicalLabel = runtimeAssets.charLabelAliases[character]?[rawLabel] ?? rawLabel
            bopomofoByChar[globalIndex] = canonicalLabel
            phoneTemplateByChar[globalIndex] = labelTemplateByLabel[canonicalLabel] ?? labelPhoneTemplates[labelIndex]
            resolvedCount += 1
        }
        return G2PWPredictionResult(
            normalizedText: normalizedText,
            bopomofoByChar: bopomofoByChar,
            phoneTemplateByChar: phoneTemplateByChar,
            queryCount: polyphonicIndices.count,
            g2pwResolvedCount: resolvedCount
        )
    }

    private static func buildDerivedAssets(
        from runtimeAssets: G2PWRuntimeAssets
    ) -> (
        charToID: [String: Int],
        polyphonicCharSet: Set<String>,
        labelPhoneTemplates: [GPTSoVITSPhoneSyllableTemplate?],
        monophonicPhoneTemplates: [String: GPTSoVITSPhoneSyllableTemplate],
        frontendLexicon: GPTSoVITSChineseFrontendLexicon?
    ) {
        GPTSoVITSRuntimeProfiler.measure("g2pw.init.runtime_assets.derive") {
            (
                charToID: Dictionary(
                    uniqueKeysWithValues: runtimeAssets.chars.enumerated().map { (index, char) in (char, index) }
                ),
                polyphonicCharSet: Set(runtimeAssets.polyphonicChars),
                labelPhoneTemplates: runtimeAssets.labelPhoneTemplates.map { $0?.toFrontendTemplate() },
                monophonicPhoneTemplates: runtimeAssets.monophonicPhoneTemplates.mapValues { $0.toFrontendTemplate() },
                frontendLexicon: runtimeAssets.frontendLexicon?.toFrontendLexicon()
            )
        }
    }

    private static func buildLabelTemplateByLabel(
        labels: [String],
        labelPhoneTemplates: [GPTSoVITSPhoneSyllableTemplate?]
    ) -> [String: GPTSoVITSPhoneSyllableTemplate] {
        GPTSoVITSRuntimeProfiler.measure("g2pw.init.label_template_map.derive") {
            Dictionary(
                uniqueKeysWithValues: zip(labels, labelPhoneTemplates).compactMap { label, template in
                    guard let template else { return nil }
                    return (label, template)
                }
            )
        }
    }

    private func predictBatch(rows: [PreparedQueryRow]) throws -> [Int] {
        guard let lastRow = rows.last else { return [] }
        var paddedRows = rows
        if paddedRows.count < manifest.runtime.shapes.batchSize {
            paddedRows.append(contentsOf: Array(repeating: lastRow, count: manifest.runtime.shapes.batchSize - paddedRows.count))
        }

        let batchSize = manifest.runtime.shapes.batchSize
        let tokenLen = manifest.runtime.shapes.tokenLen
        let labelCount = manifest.runtime.shapes.labelCount

        let inputIDs = try makeInt32Array(
            shape: [batchSize, tokenLen],
            values: paddedRows.flatMap(\.inputIDs)
        )
        let tokenTypeIDs = try makeInt32Array(
            shape: [batchSize, tokenLen],
            values: paddedRows.flatMap(\.tokenTypeIDs)
        )
        let attentionMask = try makeInt32Array(
            shape: [batchSize, tokenLen],
            values: paddedRows.flatMap(\.attentionMask)
        )
        let phonemeMask = try makeFloat32Array(
            shape: [batchSize, labelCount],
            values: paddedRows.flatMap(\.phonemeMask)
        )
        let charIDs = try makeInt32Array(shape: [batchSize], values: paddedRows.map(\.charID))
        let positionIDs = try makeInt32Array(shape: [batchSize], values: paddedRows.map(\.positionID))

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIDs),
            "token_type_ids": MLFeatureValue(multiArray: tokenTypeIDs),
            "attention_mask": MLFeatureValue(multiArray: attentionMask),
            "phoneme_mask": MLFeatureValue(multiArray: phonemeMask),
            "char_ids": MLFeatureValue(multiArray: charIDs),
            "position_ids": MLFeatureValue(multiArray: positionIDs),
        ])
        let output = try model.prediction(from: provider)
        guard let probs = output.featureValue(for: "probs")?.multiArrayValue else {
            throw NSError(domain: "G2PWCoreMLDriver", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Missing MLMultiArray output 'probs'."
            ])
        }

        var predictions = [Int]()
        predictions.reserveCapacity(rows.count)
        for rowIndex in 0..<rows.count {
            let labelIndex = argmaxRow(probs, rowIndex: rowIndex, labelCount: labelCount)
            predictions.append(labelIndex)
        }
        return predictions
    }

    private func buildQueryRow(text: String, queryID: Int) throws -> PreparedQueryRow {
        let loweredText = text.lowercased()
        let tokenization = tokenizer.tokenizeAndMap(text: loweredText)
        let truncated = truncateQueryIfNeeded(
            text: loweredText,
            queryID: queryID,
            tokenization: tokenization
        )
        let queryCharacters = Array(truncated.text)
        guard truncated.queryID >= 0, truncated.queryID < queryCharacters.count else {
            throw NSError(domain: "G2PWCoreMLDriver", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Invalid query index \(truncated.queryID) for text '\(truncated.text)'."
            ])
        }

        let queryCharacter = String(queryCharacters[truncated.queryID])
        guard let charID = charToID[queryCharacter] else {
            throw NSError(domain: "G2PWCoreMLDriver", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "Missing g2pw char id for '\(queryCharacter)'."
            ])
        }
        guard let tokenIndex = truncated.textToToken[truncated.queryID] else {
            throw NSError(domain: "G2PWCoreMLDriver", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "Failed to align query '\(queryCharacter)' to a token index."
            ])
        }

        let processedTokens = ["[CLS]"] + truncated.tokens + ["[SEP]"]
        let inputIDs = tokenizer.convertTokensToIDs(processedTokens).map(Int32.init)
        guard inputIDs.count <= manifest.runtime.shapes.tokenLen else {
            throw NSError(domain: "G2PWCoreMLDriver", code: 5, userInfo: [
                NSLocalizedDescriptionKey: "Token count \(inputIDs.count) exceeds g2pw token capacity \(manifest.runtime.shapes.tokenLen)."
            ])
        }
        let padding = manifest.runtime.shapes.tokenLen - inputIDs.count
        let padTokenID = Int32(tokenizer.tokenID(for: "[PAD]"))
        let labelCount = manifest.runtime.shapes.labelCount
        var phonemeMask = Array(repeating: Float32(0), count: labelCount)
        for labelIndex in runtimeAssets.phonemeIndicesByCharID[charID] where labelIndex < labelCount {
            phonemeMask[labelIndex] = 1
        }

        return PreparedQueryRow(
            inputIDs: inputIDs + Array(repeating: padTokenID, count: padding),
            tokenTypeIDs: Array(repeating: 0, count: manifest.runtime.shapes.tokenLen),
            attentionMask: Array(repeating: 1, count: inputIDs.count) + Array(repeating: 0, count: padding),
            phonemeMask: phonemeMask,
            charID: Int32(charID),
            positionID: Int32(tokenIndex + 1),
        )
    }

    private func truncateQueryIfNeeded(
        text: String,
        queryID: Int,
        tokenization: GPTSoVITSWordPieceTokenizationResult
    ) -> TruncatedQueryContext {
        let truncateLen = manifest.runtime.shapes.tokenLen - 2
        if tokenization.tokens.count <= truncateLen {
            return TruncatedQueryContext(
                text: text,
                queryID: queryID,
                tokens: tokenization.tokens,
                textToToken: tokenization.textToToken
            )
        }

        let tokenPosition = tokenization.textToToken[queryID] ?? 0
        var tokenStart = tokenPosition - truncateLen / 2
        var tokenEnd = tokenStart + truncateLen
        let frontExceed = -tokenStart
        let backExceed = tokenEnd - tokenization.tokens.count
        if frontExceed > 0 {
            tokenStart += frontExceed
            tokenEnd += frontExceed
        } else if backExceed > 0 {
            tokenStart -= backExceed
            tokenEnd -= backExceed
        }

        let start = tokenization.tokenToText[tokenStart].lowerBound
        let end = tokenization.tokenToText[tokenEnd - 1].upperBound
        let characters = Array(text)
        let truncatedText = String(characters[start..<end])
        let truncatedTextToToken = Array(tokenization.textToToken[start..<end]).map { tokenIndex in
            tokenIndex.map { $0 - tokenStart }
        }
        return TruncatedQueryContext(
            text: truncatedText,
            queryID: queryID - start,
            tokens: Array(tokenization.tokens[tokenStart..<tokenEnd]),
            textToToken: truncatedTextToToken
        )
    }

    private func buildPolyphonicContext(text: String, polyphonicIndices: [Int]) -> (text: String, offset: Int) {
        guard let first = polyphonicIndices.first, let last = polyphonicIndices.last else {
            return (text, 0)
        }
        let contextChars = max(runtimeAssets.polyphonicContextChars, 0)
        guard contextChars > 0 else {
            return (text, 0)
        }
        let characters = Array(text)
        let left = max(0, first - contextChars)
        let right = min(characters.count, last + contextChars + 1)
        return (String(characters[left..<right]), left)
    }

    public func makeInt32Array(shape: [Int], values: [Int32]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map(NSNumber.init(value:)), dataType: .int32)
        guard array.count == values.count else {
            throw NSError(domain: "G2PWCoreMLDriver", code: 6, userInfo: [
                NSLocalizedDescriptionKey: "Value count \(values.count) does not match shape element count \(array.count)."
            ])
        }
        for (index, value) in values.enumerated() {
            array[index] = NSNumber(value: value)
        }
        return array
    }

    public func makeFloat32Array(shape: [Int], values: [Float32]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map(NSNumber.init(value:)), dataType: .float32)
        guard array.count == values.count else {
            throw NSError(domain: "G2PWCoreMLDriver", code: 7, userInfo: [
                NSLocalizedDescriptionKey: "Value count \(values.count) does not match shape element count \(array.count)."
            ])
        }
        let pointer = array.dataPointer.bindMemory(to: Float32.self, capacity: values.count)
        for (index, value) in values.enumerated() {
            pointer[index] = value
        }
        return array
    }

    private func argmaxRow(_ probs: MLMultiArray, rowIndex: Int, labelCount: Int) -> Int {
        var bestIndex = 0
        var bestValue = Self.value(in: probs, indices: [rowIndex, 0])
        if labelCount <= 1 {
            return bestIndex
        }
        for labelIndex in 1..<labelCount {
            let value = Self.value(in: probs, indices: [rowIndex, labelIndex])
            if value > bestValue {
                bestValue = value
                bestIndex = labelIndex
            }
        }
        return bestIndex
    }

    private static func value(in array: MLMultiArray, indices: [Int]) -> Float {
        let offset = linearOffset(for: indices, in: array)
        switch array.dataType {
        case .float32:
            return array.dataPointer.bindMemory(to: Float32.self, capacity: offset + 1)[offset]
        case .double:
            return Float(array.dataPointer.bindMemory(to: Double.self, capacity: offset + 1)[offset])
        case .float16:
            return Float(array[offset].floatValue)
        default:
            return array[offset].floatValue
        }
    }

    private static func linearOffset(for indices: [Int], in array: MLMultiArray) -> Int {
        let shape = array.shape.map { Int(truncating: $0) }
        let strides = array.strides.map { Int(truncating: $0) }
        guard indices.count == shape.count else {
            return 0
        }
        var offset = 0
        for (index, stride) in zip(indices, strides) {
            offset += index * stride
        }
        return offset
    }

    private static func loadModel(at url: URL, configuration: MLModelConfiguration) throws -> MLModel {
        try GPTSoVITSCoreMLModelLoader.loadModel(at: url, configuration: configuration)
    }
}
