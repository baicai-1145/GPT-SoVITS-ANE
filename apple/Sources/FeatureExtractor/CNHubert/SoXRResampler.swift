import Darwin
import Foundation

enum GPTSoVITSSoXRResamplerError: LocalizedError {
    case libraryNotFound
    case symbolLookupFailed(String)
    case createFailed(String)
    case processFailed(String)
    case partialInputConsumed(consumed: Int, total: Int)

    var errorDescription: String? {
        switch self {
        case .libraryNotFound:
            return "libsoxr 动态库未找到。"
        case let .symbolLookupFailed(name):
            return "libsoxr 符号加载失败: \(name)"
        case let .createFailed(message):
            return "libsoxr create 失败: \(message)"
        case let .processFailed(message):
            return "libsoxr process 失败: \(message)"
        case let .partialInputConsumed(consumed, total):
            return "libsoxr 只消费了部分输入: consumed=\(consumed) total=\(total)"
        }
    }
}

private final class GPTSoVITSSoXRDynamicLibrary {
    typealias SoxrHandle = UnsafeMutableRawPointer
    typealias SoxrError = UnsafePointer<CChar>
    typealias CreateFn = @convention(c) (
        Double,
        Double,
        UInt32,
        UnsafeMutablePointer<SoxrError?>?,
        UnsafeRawPointer?,
        UnsafeRawPointer?,
        UnsafeRawPointer?
    ) -> SoxrHandle?
    typealias ProcessFn = @convention(c) (
        SoxrHandle?,
        UnsafeRawPointer?,
        Int,
        UnsafeMutablePointer<Int>?,
        UnsafeMutableRawPointer?,
        Int,
        UnsafeMutablePointer<Int>?
    ) -> SoxrError?
    typealias DeleteFn = @convention(c) (SoxrHandle?) -> Void

    let handle: UnsafeMutableRawPointer
    let soxrCreate: CreateFn
    let soxrProcess: ProcessFn
    let soxrDelete: DeleteFn

    init() throws {
        let candidatePaths = [
            ProcessInfo.processInfo.environment["GPTSOVITS_SOXR_DYLIB"],
            "/opt/homebrew/lib/libsoxr.dylib",
            "/opt/homebrew/opt/libsoxr/lib/libsoxr.dylib",
            "/usr/local/lib/libsoxr.dylib",
            "libsoxr.dylib",
            "libsoxr.0.dylib",
        ].compactMap { $0 }.filter { !$0.isEmpty }

        guard let loadedHandle = candidatePaths.compactMap({ path in dlopen(path, RTLD_NOW | RTLD_LOCAL) }).first else {
            throw GPTSoVITSSoXRResamplerError.libraryNotFound
        }
        let resolvedHandle = loadedHandle
        handle = resolvedHandle

        func loadSymbol<T>(_ name: String, as type: T.Type) throws -> T {
            guard let symbol = dlsym(resolvedHandle, name) else {
                throw GPTSoVITSSoXRResamplerError.symbolLookupFailed(name)
            }
            return unsafeBitCast(symbol, to: type)
        }

        soxrCreate = try loadSymbol("soxr_create", as: CreateFn.self)
        soxrProcess = try loadSymbol("soxr_process", as: ProcessFn.self)
        soxrDelete = try loadSymbol("soxr_delete", as: DeleteFn.self)
    }

    deinit {
        dlclose(handle)
    }
}

enum GPTSoVITSSoXRResampler {
    private static let lock = NSLock()
    private static var cachedLibrary: GPTSoVITSSoXRDynamicLibrary?

    static func isAvailable() -> Bool {
        (try? library()) != nil
    }

    static func resample(
        mono: [Float],
        sourceRate: Int,
        targetRate: Int
    ) throws -> [Float] {
        guard !mono.isEmpty else {
            return []
        }
        let runtime = try library()
        let ratio = Double(targetRate) / Double(sourceRate)
        let outputCapacity = max(Int(ceil(Double(mono.count) * ratio)) + 512, 1)
        var output = Array(repeating: Float.zero, count: outputCapacity)

        var error: GPTSoVITSSoXRDynamicLibrary.SoxrError? = nil
        guard let resampler = runtime.soxrCreate(
            Double(sourceRate),
            Double(targetRate),
            1,
            &error,
            Optional<UnsafeRawPointer>.none,
            Optional<UnsafeRawPointer>.none,
            Optional<UnsafeRawPointer>.none
        ) else {
            throw GPTSoVITSSoXRResamplerError.createFailed(errorMessage(error))
        }
        defer {
            runtime.soxrDelete(resampler)
        }

        var inputConsumed = 0
        var outputGenerated = 0
        let outputBufferCount = output.count
        let processError = mono.withUnsafeBufferPointer { inputPointer in
            output.withUnsafeMutableBufferPointer { outputPointer -> GPTSoVITSSoXRDynamicLibrary.SoxrError? in
                runtime.soxrProcess(
                    resampler,
                    inputPointer.baseAddress,
                    mono.count,
                    &inputConsumed,
                    outputPointer.baseAddress,
                    outputBufferCount,
                    &outputGenerated
                )
            }
        }
        if let processError {
            throw GPTSoVITSSoXRResamplerError.processFailed(errorMessage(processError))
        }
        if inputConsumed != mono.count {
            throw GPTSoVITSSoXRResamplerError.partialInputConsumed(consumed: inputConsumed, total: mono.count)
        }

        while outputGenerated < output.count {
            var flushed = 0
            let flushOffset = outputGenerated
            let remainingCapacity = output.count - flushOffset
            let flushError = output.withUnsafeMutableBufferPointer { outputPointer -> GPTSoVITSSoXRDynamicLibrary.SoxrError? in
                let flushBase = outputPointer.baseAddress?.advanced(by: flushOffset)
                return runtime.soxrProcess(
                    resampler,
                    Optional<UnsafeRawPointer>.none,
                    0,
                    Optional<UnsafeMutablePointer<Int>>.none,
                    flushBase,
                    remainingCapacity,
                    &flushed
                )
            }
            if let flushError {
                throw GPTSoVITSSoXRResamplerError.processFailed(errorMessage(flushError))
            }
            if flushed == 0 {
                break
            }
            outputGenerated += flushed
        }
        return Array(output.prefix(outputGenerated))
    }

    private static func library() throws -> GPTSoVITSSoXRDynamicLibrary {
        lock.lock()
        defer { lock.unlock() }
        if let cachedLibrary {
            return cachedLibrary
        }
        let loaded = try GPTSoVITSSoXRDynamicLibrary()
        cachedLibrary = loaded
        return loaded
    }

    private static func errorMessage(_ error: GPTSoVITSSoXRDynamicLibrary.SoxrError?) -> String {
        guard let error, let text = String(validatingUTF8: error) else {
            return "unknown error"
        }
        return text
    }
}
