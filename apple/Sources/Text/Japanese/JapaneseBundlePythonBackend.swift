import Foundation

public enum JapaneseBundlePythonBackendError: LocalizedError {
    case unsupportedLanguage(String)

    public var errorDescription: String? {
        switch self {
        case let .unsupportedLanguage(language):
            return "JapaneseBundlePythonBackend 只支持日语主链，收到 language=\(language)"
        }
    }
}

public final class JapaneseBundlePythonBackend: GPTSoVITSTextPhoneBackend {
    public let bundle: JapaneseFrontendBundle
    private let backend: GPTSoVITSPythonTextPhoneBackend

    public init(
        bundleDirectory: URL,
        repositoryRoot: URL,
        condaEnvironmentName: String? = GPTSoVITSPythonTextPhoneBackend.defaultCondaEnvironmentName(),
        pythonExecutable: String = ProcessInfo.processInfo.environment["GPTSOVITS_PYTHON_FRONTEND_PYTHON"] ?? "python",
        cleanerVersion: String = "v2"
    ) throws {
        let bundle = try JapaneseFrontendBundle(bundleDirectory: bundleDirectory)
        self.bundle = bundle
        self.backend = GPTSoVITSPythonTextPhoneBackend(
            repositoryRoot: repositoryRoot,
            condaEnvironmentName: condaEnvironmentName,
            pythonExecutable: pythonExecutable,
            cleanerVersion: cleanerVersion,
            environment: [
                "OPEN_JTALK_DICT_DIR": bundle.openjtalkDictionaryDirectory.path,
                "GPTSOVITS_JA_USER_DIC_DIR": bundle.jaUserDictionaryDirectory.path,
            ]
        )
    }

    public func phoneResult(
        for text: String,
        language: GPTSoVITSTextLanguage
    ) throws -> GPTSoVITSTextPhoneResult {
        guard language.baseLanguage == "ja" else {
            throw JapaneseBundlePythonBackendError.unsupportedLanguage(language.rawValue)
        }
        let result = try backend.phoneResult(for: text, language: language)
        return GPTSoVITSTextPhoneResult(
            sourceText: result.sourceText,
            normalizedText: result.normalizedText,
            phones: result.phones,
            phoneIDs: result.phoneIDs,
            word2ph: result.word2ph,
            phoneUnits: result.phoneUnits,
            backend: "python_japanese_bundle_dict"
        )
    }
}
