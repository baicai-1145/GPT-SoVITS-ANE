import CoreML
import Foundation

public extension GPTSoVITSChineseSynthesisPipeline {
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
