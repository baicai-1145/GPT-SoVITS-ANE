import Foundation

public enum KoreanMecabTaggerError: LocalizedError {
    case libraryLoadFailed(String)
    case initializationFailed(String)
    case parseFailed(String)

    public var errorDescription: String? {
        switch self {
        case let .libraryLoadFailed(message):
            return "加载 libmecab 动态库失败: \(message)"
        case let .initializationFailed(message):
            return "初始化 libmecab 失败: \(message)"
        case let .parseFailed(message):
            return "libmecab 解析失败: \(message)"
        }
    }
}

public final class KoreanMecabTagger {
    public struct Token: Equatable {
        public let surface: String
        public let tag: String
    }

    private typealias MecabNew2 = @convention(c) (UnsafePointer<CChar>?) -> OpaquePointer?
    private typealias MecabDestroy = @convention(c) (OpaquePointer?) -> Void
    private typealias MecabSparseToStr2 = @convention(c) (
        OpaquePointer?,
        UnsafePointer<CChar>?,
        Int
    ) -> UnsafePointer<CChar>?
    private typealias MecabStrError = @convention(c) (OpaquePointer?) -> UnsafePointer<CChar>?

    private let libraryHandle: UnsafeMutableRawPointer
    private let handle: OpaquePointer
    private let mecabDestroyFunction: MecabDestroy
    private let mecabSparseToStr2Function: MecabSparseToStr2
    private let mecabStrErrorFunction: MecabStrError

    public init(
        dynamicLibraryURL: URL,
        dictionaryDirectory: URL,
        mecabRCFileURL: URL
    ) throws {
        guard let libraryHandle = dlopen(dynamicLibraryURL.path, RTLD_NOW) else {
            throw KoreanMecabTaggerError.libraryLoadFailed(String(cString: dlerror()))
        }
        self.libraryHandle = libraryHandle
        let mecabNew2Function: MecabNew2 = try Self.loadSymbol(named: "mecab_new2", from: libraryHandle)
        self.mecabDestroyFunction = try Self.loadSymbol(named: "mecab_destroy", from: libraryHandle)
        self.mecabSparseToStr2Function = try Self.loadSymbol(named: "mecab_sparse_tostr2", from: libraryHandle)
        self.mecabStrErrorFunction = try Self.loadSymbol(named: "mecab_strerror", from: libraryHandle)
        let argument = "--rcfile \(mecabRCFileURL.path) --dicdir \(dictionaryDirectory.path)"
        guard let handle = argument.withCString({ mecabNew2Function($0) }) else {
            let message = Self.lastErrorMessage(using: mecabStrErrorFunction)
            dlclose(libraryHandle)
            throw KoreanMecabTaggerError.initializationFailed(message)
        }
        self.handle = handle
    }

    deinit {
        mecabDestroyFunction(handle)
        dlclose(libraryHandle)
    }

    public func pos(_ text: String) throws -> [Token] {
        guard let raw = text.withCString({ mecabSparseToStr2Function(handle, $0, strlen($0)) }) else {
            throw KoreanMecabTaggerError.parseFailed(Self.lastErrorMessage(handle: handle, using: mecabStrErrorFunction))
        }
        return String(cString: raw)
            .split(separator: "\n")
            .compactMap { line in
                guard line != "EOS", !line.isEmpty else {
                    return nil
                }
                let columns = line.split(separator: "\t", omittingEmptySubsequences: false)
                guard columns.count >= 2 else {
                    return nil
                }
                let tag = columns[1].split(separator: ",", omittingEmptySubsequences: false).first.map(String.init) ?? ""
                return Token(surface: String(columns[0]), tag: tag)
            }
    }

    private static func lastErrorMessage(
        handle: OpaquePointer? = nil,
        using mecabStrError: MecabStrError
    ) -> String {
        guard let raw = mecabStrError(handle) else {
            return "unknown mecab error"
        }
        let message = String(cString: raw)
        return message.isEmpty ? "unknown mecab error" : message
    }

    private static func loadSymbol<T>(
        named symbol: String,
        from handle: UnsafeMutableRawPointer
    ) throws -> T {
        guard let raw = dlsym(handle, symbol) else {
            let message = dlerror().map { String(cString: $0) } ?? "missing symbol \(symbol)"
            throw KoreanMecabTaggerError.libraryLoadFailed(message)
        }
        return unsafeBitCast(raw, to: T.self)
    }
}
