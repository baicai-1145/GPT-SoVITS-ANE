import CoreML
import Foundation

public enum GPTSoVITSDefaultTextPhoneBackendError: LocalizedError {
    case repositoryRootRequired(String)
    case invalidBundleDirectory(path: String, expectedBundleType: String)

    public var errorDescription: String? {
        switch self {
        case let .repositoryRootRequired(language):
            return "language=\(language) 需要 repositoryRoot 才能回退到 Python text frontend backend。"
        case let .invalidBundleDirectory(path, expectedBundleType):
            return "bundle 路径不匹配，path=\(path)，expected bundle_type=\(expectedBundleType)。"
        }
    }
}

public enum GPTSoVITSDefaultTextPhoneBackendResolver {
    public static func makeBackend(
        for language: GPTSoVITSTextLanguage,
        repositoryRoot: URL? = nil,
        englishFrontendBundleDirectory: URL? = nil,
        yueFrontendBundleDirectory: URL? = nil,
        japaneseFrontendBundleDirectory: URL? = nil,
        koreanFrontendBundleDirectory: URL? = nil,
        condaEnvironmentName: String? = GPTSoVITSPythonTextPhoneBackend.defaultCondaEnvironmentName(),
        pythonExecutable: String = ProcessInfo.processInfo.environment["GPTSOVITS_PYTHON_FRONTEND_PYTHON"] ?? "python",
        modelConfiguration: MLModelConfiguration = MLModelConfiguration()
    ) throws -> (any GPTSoVITSTextPhoneBackend)? {
        switch language.baseLanguage {
        case "zh":
            return nil
        case "en":
            if let englishFrontendBundleDirectory {
                guard EnglishCoreMLPhoneBackend.isBundleDirectory(englishFrontendBundleDirectory) else {
                    throw GPTSoVITSDefaultTextPhoneBackendError.invalidBundleDirectory(
                        path: englishFrontendBundleDirectory.path,
                        expectedBundleType: "gpt_sovits_english_frontend_bundle"
                    )
                }
                return try EnglishCoreMLPhoneBackend(
                    bundleDirectory: englishFrontendBundleDirectory,
                    configuration: modelConfiguration
                )
            }
            return try makePythonBackend(
                language: language,
                repositoryRoot: repositoryRoot,
                condaEnvironmentName: condaEnvironmentName,
                pythonExecutable: pythonExecutable
            )
        case "yue":
            if let yueFrontendBundleDirectory {
                guard CantonesePhoneFrontend.isBundleDirectory(yueFrontendBundleDirectory) else {
                    throw GPTSoVITSDefaultTextPhoneBackendError.invalidBundleDirectory(
                        path: yueFrontendBundleDirectory.path,
                        expectedBundleType: "gpt_sovits_yue_frontend_bundle"
                    )
                }
                return try CantonesePhoneFrontend(bundleDirectory: yueFrontendBundleDirectory)
            }
            return try makePythonBackend(
                language: language,
                repositoryRoot: repositoryRoot,
                condaEnvironmentName: condaEnvironmentName,
                pythonExecutable: pythonExecutable
            )
        case "ja":
            if let japaneseFrontendBundleDirectory {
                guard JapaneseFrontendBundle.isBundleDirectory(japaneseFrontendBundleDirectory) else {
                    throw GPTSoVITSDefaultTextPhoneBackendError.invalidBundleDirectory(
                        path: japaneseFrontendBundleDirectory.path,
                        expectedBundleType: "gpt_sovits_japanese_frontend_bundle"
                    )
                }
                return try JapanesePhoneFrontend(
                    bundleDirectory: japaneseFrontendBundleDirectory
                )
            }
            return try makePythonBackend(
                language: language,
                repositoryRoot: repositoryRoot,
                condaEnvironmentName: condaEnvironmentName,
                pythonExecutable: pythonExecutable
            )
        case "ko":
            if let koreanFrontendBundleDirectory {
                guard KoreanFrontendBundle.isBundleDirectory(koreanFrontendBundleDirectory) else {
                    throw GPTSoVITSDefaultTextPhoneBackendError.invalidBundleDirectory(
                        path: koreanFrontendBundleDirectory.path,
                        expectedBundleType: "gpt_sovits_korean_frontend_bundle"
                    )
                }
                return try KoreanPhoneFrontend(bundleDirectory: koreanFrontendBundleDirectory)
            }
            return try makePythonBackend(
                language: language,
                repositoryRoot: repositoryRoot,
                condaEnvironmentName: condaEnvironmentName,
                pythonExecutable: pythonExecutable
            )
        default:
            throw GPTSoVITSTextPhoneFrontendError.unsupportedLanguage(language.rawValue)
        }
    }

    private static func makePythonBackend(
        language: GPTSoVITSTextLanguage,
        repositoryRoot: URL?,
        condaEnvironmentName: String?,
        pythonExecutable: String
    ) throws -> any GPTSoVITSTextPhoneBackend {
        guard let repositoryRoot else {
            throw GPTSoVITSDefaultTextPhoneBackendError.repositoryRootRequired(language.rawValue)
        }
        return GPTSoVITSPythonTextPhoneBackend(
            repositoryRoot: repositoryRoot,
            condaEnvironmentName: condaEnvironmentName,
            pythonExecutable: pythonExecutable
        )
    }
}

public extension GPTSoVITST2STextPreparer {
    func resolveDefaultTextPhoneBackend(
        for language: GPTSoVITSTextLanguage,
        repositoryRoot: URL? = nil,
        englishFrontendBundleDirectory: URL? = nil,
        yueFrontendBundleDirectory: URL? = nil,
        japaneseFrontendBundleDirectory: URL? = nil,
        koreanFrontendBundleDirectory: URL? = nil,
        condaEnvironmentName: String? = GPTSoVITSPythonTextPhoneBackend.defaultCondaEnvironmentName(),
        pythonExecutable: String = ProcessInfo.processInfo.environment["GPTSOVITS_PYTHON_FRONTEND_PYTHON"] ?? "python",
        modelConfiguration: MLModelConfiguration = MLModelConfiguration()
    ) throws {
        textPhoneBackend = try GPTSoVITSDefaultTextPhoneBackendResolver.makeBackend(
            for: language,
            repositoryRoot: repositoryRoot,
            englishFrontendBundleDirectory: englishFrontendBundleDirectory,
            yueFrontendBundleDirectory: yueFrontendBundleDirectory,
            japaneseFrontendBundleDirectory: japaneseFrontendBundleDirectory,
            koreanFrontendBundleDirectory: koreanFrontendBundleDirectory,
            condaEnvironmentName: condaEnvironmentName,
            pythonExecutable: pythonExecutable,
            modelConfiguration: modelConfiguration
        )
    }
}
