import Foundation

public enum KoreanBundlePythonBackendError: LocalizedError {
    case unsupportedLanguage(String)

    public var errorDescription: String? {
        switch self {
        case let .unsupportedLanguage(language):
            return "KoreanBundlePythonBackend 只支持韩语主链，收到 language=\(language)"
        }
    }
}

public final class KoreanBundlePythonBackend: GPTSoVITSTextPhoneBackend {
    public let bundle: KoreanFrontendBundle
    private let backend: GPTSoVITSPythonTextPhoneBackend

    public init(
        bundleDirectory: URL,
        repositoryRoot: URL,
        condaEnvironmentName: String? = GPTSoVITSPythonTextPhoneBackend.defaultCondaEnvironmentName(),
        pythonExecutable: String = ProcessInfo.processInfo.environment["GPTSOVITS_PYTHON_FRONTEND_PYTHON"] ?? "python",
        cleanerVersion: String = "v2"
    ) throws {
        let bundle = try KoreanFrontendBundle(bundleDirectory: bundleDirectory)
        self.bundle = bundle
        self.backend = GPTSoVITSPythonTextPhoneBackend(
            repositoryRoot: repositoryRoot,
            condaEnvironmentName: condaEnvironmentName,
            pythonExecutable: pythonExecutable,
            cleanerVersion: cleanerVersion,
            environment: [
                "GPTSOVITS_MECAB_KO_DIC_DIR": bundle.mecabDictionaryDirectory.path,
            ]
        )
    }

    public func phoneResult(
        for text: String,
        language: GPTSoVITSTextLanguage
    ) throws -> GPTSoVITSTextPhoneResult {
        guard language.baseLanguage == "ko" else {
            throw KoreanBundlePythonBackendError.unsupportedLanguage(language.rawValue)
        }
        let result = try backend.phoneResult(for: text, language: language)
        return GPTSoVITSTextPhoneResult(
            sourceText: result.sourceText,
            normalizedText: result.normalizedText,
            phones: result.phones,
            phoneIDs: result.phoneIDs,
            word2ph: result.word2ph,
            phoneUnits: result.phoneUnits,
            backend: "python_korean_bundle_dict"
        )
    }
}
