import Foundation

public enum JapaneseOpenJTalkError: LocalizedError {
    case libraryLoadFailed(String)
    case symbolMissing(String)
    case initializationFailed(String)
    case mecabLoadFailed(String)
    case analysisFailed(String)
    case userDictionaryBuildFailed(String)

    public var errorDescription: String? {
        switch self {
        case let .libraryLoadFailed(message):
            return "加载日语 OpenJTalk 动态库失败: \(message)"
        case let .symbolMissing(symbol):
            return "日语 OpenJTalk 动态库缺少符号: \(symbol)"
        case let .initializationFailed(message):
            return "初始化日语 OpenJTalk 失败: \(message)"
        case let .mecabLoadFailed(message):
            return "加载日语 MeCab/OpenJTalk 词典失败: \(message)"
        case let .analysisFailed(message):
            return "日语 OpenJTalk 分析失败: \(message)"
        case let .userDictionaryBuildFailed(message):
            return "构建日语用户词典失败: \(message)"
        }
    }
}

public final class JapaneseOpenJTalk {
    public struct Feature: Equatable {
        public let string: String
        public let pos: String
        public let posGroup1: String
        public let posGroup2: String
        public let posGroup3: String
        public let ctype: String
        public let cform: String
        public let orig: String
        public let read: String
        public let pron: String
        public let acc: Int
        public let moraSize: Int
        public let chainRule: String
        public let chainFlag: Int
    }

    private struct MecabNative {
        var feature: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
        var size: Int32 = 0
        var model: UnsafeMutableRawPointer?
        var tagger: UnsafeMutableRawPointer?
        var lattice: UnsafeMutableRawPointer?
    }

    private struct NJDNodeNative {
        var string: UnsafeMutablePointer<CChar>?
        var pos: UnsafeMutablePointer<CChar>?
        var posGroup1: UnsafeMutablePointer<CChar>?
        var posGroup2: UnsafeMutablePointer<CChar>?
        var posGroup3: UnsafeMutablePointer<CChar>?
        var ctype: UnsafeMutablePointer<CChar>?
        var cform: UnsafeMutablePointer<CChar>?
        var orig: UnsafeMutablePointer<CChar>?
        var read: UnsafeMutablePointer<CChar>?
        var pron: UnsafeMutablePointer<CChar>?
        var acc: Int32 = 0
        var moraSize: Int32 = 0
        var chainRule: UnsafeMutablePointer<CChar>?
        var chainFlag: Int32 = 0
        var prev: UnsafeMutablePointer<NJDNodeNative>?
        var next: UnsafeMutablePointer<NJDNodeNative>?
    }

    private struct NJDNative {
        var head: UnsafeMutablePointer<NJDNodeNative>?
        var tail: UnsafeMutablePointer<NJDNodeNative>?
    }

    private struct JPCommonNative {
        var head: UnsafeMutableRawPointer?
        var tail: UnsafeMutableRawPointer?
        var label: UnsafeMutableRawPointer?
    }

    private typealias Text2Mecab = @convention(c) (UnsafeMutablePointer<CChar>?, UnsafePointer<CChar>?) -> Void
    private typealias MecabInitialize = @convention(c) (UnsafeMutableRawPointer?) -> Int32
    private typealias MecabLoad = @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<CChar>?) -> Int32
    private typealias MecabAnalysis = @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<CChar>?) -> Int32
    private typealias MecabGetFeature = @convention(c) (UnsafeMutableRawPointer?) -> UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
    private typealias MecabGetSize = @convention(c) (UnsafeMutableRawPointer?) -> Int32
    private typealias MecabRefresh = @convention(c) (UnsafeMutableRawPointer?) -> Int32
    private typealias MecabClear = @convention(c) (UnsafeMutableRawPointer?) -> Int32
    private typealias MecabModelNew2 = @convention(c) (UnsafePointer<CChar>?) -> OpaquePointer?
    private typealias MecabModelDestroy = @convention(c) (OpaquePointer?) -> Void
    private typealias MecabModelNewTagger = @convention(c) (OpaquePointer?) -> OpaquePointer?
    private typealias MecabModelNewLattice = @convention(c) (OpaquePointer?) -> OpaquePointer?
    private typealias MecabDictIndex = @convention(c) (Int32, UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?) -> Int32
    private typealias Mecab2NJD = @convention(c) (
        UnsafeMutableRawPointer?,
        UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?,
        Int32
    ) -> Void
    private typealias NJDInitialize = @convention(c) (UnsafeMutableRawPointer?) -> Void
    private typealias NJDRefresh = @convention(c) (UnsafeMutableRawPointer?) -> Void
    private typealias NJDClear = @convention(c) (UnsafeMutableRawPointer?) -> Void
    private typealias NJDSetTransform = @convention(c) (UnsafeMutableRawPointer?) -> Void
    private typealias NJD2JPCommon = @convention(c) (UnsafeMutableRawPointer?, UnsafeMutableRawPointer?) -> Void
    private typealias NJDNodeGetterString = @convention(c) (UnsafeMutableRawPointer?) -> UnsafePointer<CChar>?
    private typealias NJDNodeGetterInt = @convention(c) (UnsafeMutableRawPointer?) -> Int32
    private typealias JPCommonInitialize = @convention(c) (UnsafeMutableRawPointer?) -> Void
    private typealias JPCommonMakeLabel = @convention(c) (UnsafeMutableRawPointer?) -> Void
    private typealias JPCommonGetLabelSize = @convention(c) (UnsafeMutableRawPointer?) -> Int32
    private typealias JPCommonGetLabelFeature = @convention(c) (
        UnsafeMutableRawPointer?
    ) -> UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
    private typealias JPCommonRefresh = @convention(c) (UnsafeMutableRawPointer?) -> Void
    private typealias JPCommonClear = @convention(c) (UnsafeMutableRawPointer?) -> Void

    private let libraryHandle: UnsafeMutableRawPointer
    private var mecab = MecabNative()
    private var njd = NJDNative()
    private var jpcommon = JPCommonNative()

    private let text2mecabFunction: Text2Mecab
    private let mecabInitializeFunction: MecabInitialize
    private let mecabLoadFunction: MecabLoad
    private let mecabAnalysisFunction: MecabAnalysis
    private let mecabGetFeatureFunction: MecabGetFeature
    private let mecabGetSizeFunction: MecabGetSize
    private let mecabRefreshFunction: MecabRefresh
    private let mecabClearFunction: MecabClear
    private let mecabModelNew2Function: MecabModelNew2
    private let mecabModelDestroyFunction: MecabModelDestroy
    private let mecabModelNewTaggerFunction: MecabModelNewTagger
    private let mecabModelNewLatticeFunction: MecabModelNewLattice
    private let mecabDictIndexFunction: MecabDictIndex
    private let mecab2njdFunction: Mecab2NJD
    private let njdInitializeFunction: NJDInitialize
    private let njdRefreshFunction: NJDRefresh
    private let njdClearFunction: NJDClear
    private let njdSetPronunciationFunction: NJDSetTransform
    private let njdSetDigitFunction: NJDSetTransform
    private let njdSetAccentPhraseFunction: NJDSetTransform
    private let njdSetAccentTypeFunction: NJDSetTransform
    private let njdSetUnvoicedVowelFunction: NJDSetTransform
    private let njdSetLongVowelFunction: NJDSetTransform
    private let njd2jpcommonFunction: NJD2JPCommon
    private let njdNodeGetStringFunction: NJDNodeGetterString
    private let njdNodeGetPOSFunction: NJDNodeGetterString
    private let njdNodeGetPOSGroup1Function: NJDNodeGetterString
    private let njdNodeGetPOSGroup2Function: NJDNodeGetterString
    private let njdNodeGetPOSGroup3Function: NJDNodeGetterString
    private let njdNodeGetCTypeFunction: NJDNodeGetterString
    private let njdNodeGetCFormFunction: NJDNodeGetterString
    private let njdNodeGetOrigFunction: NJDNodeGetterString
    private let njdNodeGetReadFunction: NJDNodeGetterString
    private let njdNodeGetPronFunction: NJDNodeGetterString
    private let njdNodeGetAccFunction: NJDNodeGetterInt
    private let njdNodeGetMoraSizeFunction: NJDNodeGetterInt
    private let njdNodeGetChainRuleFunction: NJDNodeGetterString
    private let njdNodeGetChainFlagFunction: NJDNodeGetterInt
    private let jpcommonInitializeFunction: JPCommonInitialize
    private let jpcommonMakeLabelFunction: JPCommonMakeLabel
    private let jpcommonGetLabelSizeFunction: JPCommonGetLabelSize
    private let jpcommonGetLabelFeatureFunction: JPCommonGetLabelFeature
    private let jpcommonRefreshFunction: JPCommonRefresh
    private let jpcommonClearFunction: JPCommonClear

    public init(
        dynamicLibraryURL: URL,
        dictionaryDirectory: URL,
        userDictionaryDirectory: URL? = nil
    ) throws {
        guard let libraryHandle = dlopen(dynamicLibraryURL.path, RTLD_NOW) else {
            throw JapaneseOpenJTalkError.libraryLoadFailed(String(cString: dlerror()))
        }
        self.libraryHandle = libraryHandle

        self.text2mecabFunction = try Self.loadSymbol(named: "text2mecab", from: libraryHandle)
        self.mecabInitializeFunction = try Self.loadSymbol(named: "Mecab_initialize", from: libraryHandle)
        self.mecabLoadFunction = try Self.loadSymbol(named: "Mecab_load", from: libraryHandle)
        self.mecabAnalysisFunction = try Self.loadSymbol(named: "Mecab_analysis", from: libraryHandle)
        self.mecabGetFeatureFunction = try Self.loadSymbol(named: "Mecab_get_feature", from: libraryHandle)
        self.mecabGetSizeFunction = try Self.loadSymbol(named: "Mecab_get_size", from: libraryHandle)
        self.mecabRefreshFunction = try Self.loadSymbol(named: "Mecab_refresh", from: libraryHandle)
        self.mecabClearFunction = try Self.loadSymbol(named: "Mecab_clear", from: libraryHandle)
        self.mecabModelNew2Function = try Self.loadSymbol(named: "mecab_model_new2", from: libraryHandle)
        self.mecabModelDestroyFunction = try Self.loadSymbol(named: "mecab_model_destroy", from: libraryHandle)
        self.mecabModelNewTaggerFunction = try Self.loadSymbol(named: "mecab_model_new_tagger", from: libraryHandle)
        self.mecabModelNewLatticeFunction = try Self.loadSymbol(named: "mecab_model_new_lattice", from: libraryHandle)
        self.mecabDictIndexFunction = try Self.loadSymbol(named: "mecab_dict_index", from: libraryHandle)
        self.mecab2njdFunction = try Self.loadSymbol(named: "mecab2njd", from: libraryHandle)
        self.njdInitializeFunction = try Self.loadSymbol(named: "NJD_initialize", from: libraryHandle)
        self.njdRefreshFunction = try Self.loadSymbol(named: "NJD_refresh", from: libraryHandle)
        self.njdClearFunction = try Self.loadSymbol(named: "NJD_clear", from: libraryHandle)
        self.njdSetPronunciationFunction = try Self.loadSymbol(named: "njd_set_pronunciation", from: libraryHandle)
        self.njdSetDigitFunction = try Self.loadSymbol(named: "njd_set_digit", from: libraryHandle)
        self.njdSetAccentPhraseFunction = try Self.loadSymbol(named: "njd_set_accent_phrase", from: libraryHandle)
        self.njdSetAccentTypeFunction = try Self.loadSymbol(named: "njd_set_accent_type", from: libraryHandle)
        self.njdSetUnvoicedVowelFunction = try Self.loadSymbol(named: "njd_set_unvoiced_vowel", from: libraryHandle)
        self.njdSetLongVowelFunction = try Self.loadSymbol(named: "njd_set_long_vowel", from: libraryHandle)
        self.njd2jpcommonFunction = try Self.loadSymbol(named: "njd2jpcommon", from: libraryHandle)
        self.njdNodeGetStringFunction = try Self.loadSymbol(named: "NJDNode_get_string", from: libraryHandle)
        self.njdNodeGetPOSFunction = try Self.loadSymbol(named: "NJDNode_get_pos", from: libraryHandle)
        self.njdNodeGetPOSGroup1Function = try Self.loadSymbol(named: "NJDNode_get_pos_group1", from: libraryHandle)
        self.njdNodeGetPOSGroup2Function = try Self.loadSymbol(named: "NJDNode_get_pos_group2", from: libraryHandle)
        self.njdNodeGetPOSGroup3Function = try Self.loadSymbol(named: "NJDNode_get_pos_group3", from: libraryHandle)
        self.njdNodeGetCTypeFunction = try Self.loadSymbol(named: "NJDNode_get_ctype", from: libraryHandle)
        self.njdNodeGetCFormFunction = try Self.loadSymbol(named: "NJDNode_get_cform", from: libraryHandle)
        self.njdNodeGetOrigFunction = try Self.loadSymbol(named: "NJDNode_get_orig", from: libraryHandle)
        self.njdNodeGetReadFunction = try Self.loadSymbol(named: "NJDNode_get_read", from: libraryHandle)
        self.njdNodeGetPronFunction = try Self.loadSymbol(named: "NJDNode_get_pron", from: libraryHandle)
        self.njdNodeGetAccFunction = try Self.loadSymbol(named: "NJDNode_get_acc", from: libraryHandle)
        self.njdNodeGetMoraSizeFunction = try Self.loadSymbol(named: "NJDNode_get_mora_size", from: libraryHandle)
        self.njdNodeGetChainRuleFunction = try Self.loadSymbol(named: "NJDNode_get_chain_rule", from: libraryHandle)
        self.njdNodeGetChainFlagFunction = try Self.loadSymbol(named: "NJDNode_get_chain_flag", from: libraryHandle)
        self.jpcommonInitializeFunction = try Self.loadSymbol(named: "JPCommon_initialize", from: libraryHandle)
        self.jpcommonMakeLabelFunction = try Self.loadSymbol(named: "JPCommon_make_label", from: libraryHandle)
        self.jpcommonGetLabelSizeFunction = try Self.loadSymbol(named: "JPCommon_get_label_size", from: libraryHandle)
        self.jpcommonGetLabelFeatureFunction = try Self.loadSymbol(named: "JPCommon_get_label_feature", from: libraryHandle)
        self.jpcommonRefreshFunction = try Self.loadSymbol(named: "JPCommon_refresh", from: libraryHandle)
        self.jpcommonClearFunction = try Self.loadSymbol(named: "JPCommon_clear", from: libraryHandle)

        guard withUnsafeMutablePointer(to: &mecab, { mecabInitializeFunction(UnsafeMutableRawPointer($0)) }) == 1 else {
            dlclose(libraryHandle)
            throw JapaneseOpenJTalkError.initializationFailed("Mecab_initialize returned false")
        }
        withUnsafeMutablePointer(to: &njd) { njdInitializeFunction(UnsafeMutableRawPointer($0)) }
        withUnsafeMutablePointer(to: &jpcommon) { jpcommonInitializeFunction(UnsafeMutableRawPointer($0)) }

        let userDictionaryURL = try Self.prepareUserDictionary(
            userDictionaryDirectory: userDictionaryDirectory,
            dictionaryDirectory: dictionaryDirectory,
            mecabDictIndexFunction: mecabDictIndexFunction
        )
        do {
            try loadMecab(
                dictionaryDirectory: dictionaryDirectory,
                userDictionaryURL: userDictionaryURL
            )
        } catch {
            _ = withUnsafeMutablePointer(to: &mecab) { mecabClearFunction(UnsafeMutableRawPointer($0)) }
            withUnsafeMutablePointer(to: &njd) { njdClearFunction(UnsafeMutableRawPointer($0)) }
            withUnsafeMutablePointer(to: &jpcommon) { jpcommonClearFunction(UnsafeMutableRawPointer($0)) }
            dlclose(libraryHandle)
            throw error
        }
    }

    deinit {
        _ = withUnsafeMutablePointer(to: &mecab) { mecabClearFunction(UnsafeMutableRawPointer($0)) }
        withUnsafeMutablePointer(to: &njd) { njdClearFunction(UnsafeMutableRawPointer($0)) }
        withUnsafeMutablePointer(to: &jpcommon) { jpcommonClearFunction(UnsafeMutableRawPointer($0)) }
        dlclose(libraryHandle)
    }

    public func analyze(_ text: String) throws -> (features: [Feature], labels: [String]) {
        let preparedText = text.precomposedStringWithCanonicalMapping
        var convertedBuffer = [CChar](repeating: 0, count: max(8192, preparedText.utf8.count * 16 + 1024))
        let features: [Feature]
        let labels: [String]

        defer {
            withUnsafeMutablePointer(to: &jpcommon) { jpcommonRefreshFunction(UnsafeMutableRawPointer($0)) }
            withUnsafeMutablePointer(to: &njd) { njdRefreshFunction(UnsafeMutableRawPointer($0)) }
            _ = withUnsafeMutablePointer(to: &mecab) { mecabRefreshFunction(UnsafeMutableRawPointer($0)) }
        }

        convertedBuffer.withUnsafeMutableBufferPointer { buffer in
            preparedText.withCString { input in
                text2mecabFunction(buffer.baseAddress, input)
            }
        }

        let analysisResult = withUnsafeMutablePointer(to: &mecab) {
            mecabAnalysisFunction(UnsafeMutableRawPointer($0), convertedBuffer)
        }
        guard analysisResult == 1 else {
            throw JapaneseOpenJTalkError.analysisFailed("Mecab_analysis returned false")
        }
        guard let rawFeatures = withUnsafeMutablePointer(to: &mecab, {
            mecabGetFeatureFunction(UnsafeMutableRawPointer($0))
        }) else {
            throw JapaneseOpenJTalkError.analysisFailed("Mecab_get_feature returned nil")
        }
        let featureCount = withUnsafeMutablePointer(to: &mecab) {
            mecabGetSizeFunction(UnsafeMutableRawPointer($0))
        }
        withUnsafeMutablePointer(to: &njd) { njdPointer in
            let rawNJD = UnsafeMutableRawPointer(njdPointer)
            mecab2njdFunction(rawNJD, rawFeatures, featureCount)
            njdSetPronunciationFunction(rawNJD)
            njdSetDigitFunction(rawNJD)
            njdSetAccentPhraseFunction(rawNJD)
            njdSetAccentTypeFunction(rawNJD)
            njdSetUnvoicedVowelFunction(rawNJD)
            njdSetLongVowelFunction(rawNJD)
        }

        features = extractFeatures()
        withUnsafeMutablePointer(to: &jpcommon) { jpcommonPointer in
            withUnsafeMutablePointer(to: &njd) { njdPointer in
                njd2jpcommonFunction(UnsafeMutableRawPointer(jpcommonPointer), UnsafeMutableRawPointer(njdPointer))
            }
            jpcommonMakeLabelFunction(UnsafeMutableRawPointer(jpcommonPointer))
        }
        labels = extractLabels()

        return (features, labels)
    }

    private func extractFeatures() -> [Feature] {
        var result = [Feature]()
        var node = njd.head
        while let current = node {
            result.append(
                Feature(
                    string: Self.stringValue(from: njdNodeGetStringFunction(UnsafeMutableRawPointer(current))),
                    pos: Self.stringValue(from: njdNodeGetPOSFunction(UnsafeMutableRawPointer(current))),
                    posGroup1: Self.stringValue(from: njdNodeGetPOSGroup1Function(UnsafeMutableRawPointer(current))),
                    posGroup2: Self.stringValue(from: njdNodeGetPOSGroup2Function(UnsafeMutableRawPointer(current))),
                    posGroup3: Self.stringValue(from: njdNodeGetPOSGroup3Function(UnsafeMutableRawPointer(current))),
                    ctype: Self.stringValue(from: njdNodeGetCTypeFunction(UnsafeMutableRawPointer(current))),
                    cform: Self.stringValue(from: njdNodeGetCFormFunction(UnsafeMutableRawPointer(current))),
                    orig: Self.stringValue(from: njdNodeGetOrigFunction(UnsafeMutableRawPointer(current))),
                    read: Self.stringValue(from: njdNodeGetReadFunction(UnsafeMutableRawPointer(current))),
                    pron: Self.stringValue(from: njdNodeGetPronFunction(UnsafeMutableRawPointer(current))),
                    acc: Int(njdNodeGetAccFunction(UnsafeMutableRawPointer(current))),
                    moraSize: Int(njdNodeGetMoraSizeFunction(UnsafeMutableRawPointer(current))),
                    chainRule: Self.stringValue(from: njdNodeGetChainRuleFunction(UnsafeMutableRawPointer(current))),
                    chainFlag: Int(njdNodeGetChainFlagFunction(UnsafeMutableRawPointer(current)))
                )
            )
            node = current.pointee.next
        }
        return result
    }

    private func extractLabels() -> [String] {
        let count = withUnsafeMutablePointer(to: &jpcommon) {
            Int(jpcommonGetLabelSizeFunction(UnsafeMutableRawPointer($0)))
        }
        guard count > 0,
              let base = withUnsafeMutablePointer(to: &jpcommon, {
                  jpcommonGetLabelFeatureFunction(UnsafeMutableRawPointer($0))
              }) else {
            return []
        }
        var labels = [String]()
        labels.reserveCapacity(count)
        for index in 0..<count {
            labels.append(Self.stringValue(from: base[index]))
        }
        return labels
    }

    private func loadMecab(
        dictionaryDirectory: URL,
        userDictionaryURL: URL?
    ) throws {
        if let userDictionaryURL {
            _ = withUnsafeMutablePointer(to: &mecab) { mecabClearFunction(UnsafeMutableRawPointer($0)) }
            let argument = "mecab -d \(dictionaryDirectory.path) -u \(userDictionaryURL.path)"
            guard let model = argument.withCString({ mecabModelNew2Function($0) }) else {
                throw JapaneseOpenJTalkError.mecabLoadFailed(argument)
            }
            guard let tagger = mecabModelNewTaggerFunction(model) else {
                mecabModelDestroyFunction(model)
                throw JapaneseOpenJTalkError.mecabLoadFailed("mecab_model_new_tagger failed")
            }
            guard let lattice = mecabModelNewLatticeFunction(model) else {
                mecabModelDestroyFunction(model)
                throw JapaneseOpenJTalkError.mecabLoadFailed("mecab_model_new_lattice failed")
            }
            mecab.model = UnsafeMutableRawPointer(model)
            mecab.tagger = UnsafeMutableRawPointer(tagger)
            mecab.lattice = UnsafeMutableRawPointer(lattice)
            return
        }

        let result = withUnsafeMutablePointer(to: &mecab) { mecabPointer in
            dictionaryDirectory.path.withCString { mecabLoadFunction(UnsafeMutableRawPointer(mecabPointer), $0) }
        }
        guard result == 1 else {
            throw JapaneseOpenJTalkError.mecabLoadFailed(dictionaryDirectory.path)
        }
    }

    private static func prepareUserDictionary(
        userDictionaryDirectory: URL?,
        dictionaryDirectory: URL,
        mecabDictIndexFunction: MecabDictIndex
    ) throws -> URL? {
        guard let userDictionaryDirectory else {
            return nil
        }
        let binaryURL = userDictionaryDirectory.appendingPathComponent("user.dict", isDirectory: false)
        if FileManager.default.fileExists(atPath: binaryURL.path) {
            return binaryURL
        }
        let csvURL = userDictionaryDirectory.appendingPathComponent("userdict.csv", isDirectory: false)
        guard FileManager.default.fileExists(atPath: csvURL.path) else {
            return nil
        }
        let argumentStrings = [
            "mecab-dict-index",
            "-d",
            dictionaryDirectory.path,
            "-u",
            binaryURL.path,
            "-f",
            "utf-8",
            "-t",
            "utf-8",
            csvURL.path,
        ]
        var arguments = argumentStrings.map { strdup($0) }
        defer {
            for pointer in arguments {
                free(pointer)
            }
        }
        let result = mecabDictIndexFunction(Int32(arguments.count), &arguments)
        guard result == 0 else {
            throw JapaneseOpenJTalkError.userDictionaryBuildFailed(csvURL.path)
        }
        return binaryURL
    }

    private static func stringValue(from pointer: UnsafePointer<CChar>?) -> String {
        guard let pointer else { return "" }
        return String(cString: pointer)
    }

    private static func loadSymbol<T>(
        named symbol: String,
        from handle: UnsafeMutableRawPointer
    ) throws -> T {
        guard let raw = dlsym(handle, symbol) else {
            throw JapaneseOpenJTalkError.symbolMissing(symbol)
        }
        return unsafeBitCast(raw, to: T.self)
    }
}
