import Foundation

public enum GPTSoVITSPythonTextPhoneBackendError: LocalizedError {
    case invalidRepositoryRoot(String)
    case processFailed(command: [String], code: Int32, stderr: String)
    case invalidPayload(String)

    public var errorDescription: String? {
        switch self {
        case let .invalidRepositoryRoot(path):
            return "Python 文本前端桥接找不到仓库目录: \(path)"
        case let .processFailed(command, code, stderr):
            let commandLine = command.joined(separator: " ")
            let stderrText = stderr.isEmpty ? "<empty>" : stderr
            return "Python 文本前端桥接执行失败 code=\(code): \(commandLine)\nstderr: \(stderrText)"
        case let .invalidPayload(message):
            return "Python 文本前端桥接返回了无效 payload: \(message)"
        }
    }
}

public final class GPTSoVITSPythonTextPhoneBackend: GPTSoVITSTextPhoneBackend {
    public let repositoryRoot: URL
    public let scriptPath: String
    public let condaEnvironmentName: String?
    public let pythonExecutable: String
    public let cleanerVersion: String
    public let environment: [String: String]

    public init(
        repositoryRoot: URL,
        scriptPath: String = "GPT_SoVITS/export_coreml/apple_text_frontend_cli.py",
        condaEnvironmentName: String? = GPTSoVITSPythonTextPhoneBackend.defaultCondaEnvironmentName(),
        pythonExecutable: String = ProcessInfo.processInfo.environment["GPTSOVITS_PYTHON_FRONTEND_PYTHON"] ?? "python",
        cleanerVersion: String = "v2",
        environment: [String: String] = [:]
    ) {
        self.repositoryRoot = repositoryRoot
        self.scriptPath = scriptPath
        self.condaEnvironmentName = condaEnvironmentName
        self.pythonExecutable = pythonExecutable
        self.cleanerVersion = cleanerVersion
        self.environment = environment
    }

    public func phoneResult(
        for text: String,
        language: GPTSoVITSTextLanguage
    ) throws -> GPTSoVITSTextPhoneResult {
        let rootPath = repositoryRoot.standardizedFileURL.path
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: rootPath, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw GPTSoVITSPythonTextPhoneBackendError.invalidRepositoryRoot(rootPath)
        }

        let payload = try runCLI(text: text, language: language)
        let jsonPayload = try normalizedJSONPayload(from: payload)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let decoded: BackendPhoneResult
        do {
            decoded = try decoder.decode(BackendPhoneResult.self, from: jsonPayload)
        } catch {
            let raw = String(decoding: payload, as: UTF8.self)
            throw GPTSoVITSPythonTextPhoneBackendError.invalidPayload("\(error)\nraw=\(raw)")
        }
        return try decoded.toFrontendResult()
    }

    private func runCLI(text: String, language: GPTSoVITSTextLanguage) throws -> Data {
        let scriptURL = URL(fileURLWithPath: scriptPath, relativeTo: repositoryRoot)
        var command = [String]()
        let process = Process()
        process.currentDirectoryURL = repositoryRoot
        var processEnvironment = ProcessInfo.processInfo.environment
        for (key, value) in environment where !value.isEmpty {
            processEnvironment[key] = value
        }
        process.environment = processEnvironment

        if let condaEnvironmentName, !condaEnvironmentName.isEmpty {
            process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            process.arguments = [
                "conda",
                "run",
                "-n",
                condaEnvironmentName,
                pythonExecutable,
                scriptURL.path,
                "--text",
                text,
                "--language",
                language.rawValue,
                "--version",
                cleanerVersion,
            ]
            command = process.arguments ?? []
        } else {
            process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            process.arguments = [
                pythonExecutable,
                scriptURL.path,
                "--text",
                text,
                "--language",
                language.rawValue,
                "--version",
                cleanerVersion,
            ]
            command = process.arguments ?? []
        }

        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        process.waitUntilExit()

        let stdoutData = stdout.fileHandleForReading.readDataToEndOfFile()
        let stderrData = stderr.fileHandleForReading.readDataToEndOfFile()
        guard process.terminationStatus == 0 else {
            throw GPTSoVITSPythonTextPhoneBackendError.processFailed(
                command: command,
                code: process.terminationStatus,
                stderr: String(decoding: stderrData, as: UTF8.self)
            )
        }
        return stdoutData
    }

    private func normalizedJSONPayload(from payload: Data) throws -> Data {
        if (try? JSONSerialization.jsonObject(with: payload)) != nil {
            return payload
        }
        let raw = String(decoding: payload, as: UTF8.self)
        guard let start = raw.firstIndex(of: "{"), let end = raw.lastIndex(of: "}") else {
            throw GPTSoVITSPythonTextPhoneBackendError.invalidPayload(raw)
        }
        let slice = raw[start...end]
        let data = Data(slice.utf8)
        guard (try? JSONSerialization.jsonObject(with: data)) != nil else {
            throw GPTSoVITSPythonTextPhoneBackendError.invalidPayload(raw)
        }
        return data
    }

    public static func defaultCondaEnvironmentName() -> String? {
        let configured = ProcessInfo.processInfo.environment["GPTSOVITS_PYTHON_FRONTEND_ENV"]
        if let configured {
            return configured.isEmpty ? nil : configured
        }
        return "GPTSoVits"
    }
}

private struct BackendPhoneResult: Decodable {
    let sourceText: String
    let normalizedText: String
    let phones: [String]
    let phoneIds: [Int]
    let word2ph: [Int]?
    let phoneUnits: [BackendPhoneUnit]?
    let backend: String

    func toFrontendResult() throws -> GPTSoVITSTextPhoneResult {
        GPTSoVITSTextPhoneResult(
            sourceText: sourceText.precomposedStringWithCanonicalMapping,
            normalizedText: normalizedText.precomposedStringWithCanonicalMapping,
            phones: phones,
            phoneIDs: phoneIds,
            word2ph: word2ph,
            phoneUnits: try (phoneUnits ?? []).map { try $0.toFrontendUnit() },
            backend: backend
        )
    }
}

private struct BackendPhoneUnit: Decodable {
    let unitType: String
    let text: String
    let normText: String
    let pos: String?
    let phones: [String]
    let phoneIds: [Int]
    let charStart: Int
    let charEnd: Int
    let phoneStart: Int
    let phoneEnd: Int
    let phoneCount: Int

    func toFrontendUnit() throws -> GPTSoVITSTextPhoneUnit {
        guard let mappedType = GPTSoVITSTextPhoneUnitType(rawValue: unitType) else {
            throw GPTSoVITSPythonTextPhoneBackendError.invalidPayload("unknown unit_type=\(unitType)")
        }
        return GPTSoVITSTextPhoneUnit(
            unitType: mappedType,
            text: text.precomposedStringWithCanonicalMapping,
            normText: normText.precomposedStringWithCanonicalMapping,
            pos: pos?.precomposedStringWithCanonicalMapping,
            phones: phones,
            phoneIDs: phoneIds,
            charStart: charStart,
            charEnd: charEnd,
            phoneStart: phoneStart,
            phoneEnd: phoneEnd,
            phoneCount: phoneCount
        )
    }
}
