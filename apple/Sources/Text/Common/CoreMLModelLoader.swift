import CoreML
import Foundation

public enum GPTSoVITSCoreMLModelLoader {
    public static func loadModel(
        at url: URL,
        configuration: MLModelConfiguration = MLModelConfiguration()
    ) throws -> MLModel {
        let metricBase = metricBaseName(for: url)
        let loadURL = try GPTSoVITSRuntimeProfiler.measure("\(metricBase).resolve") {
            try resolvedModelURL(for: url)
        }
        return try GPTSoVITSRuntimeProfiler.measure("\(metricBase).load") {
            try MLModel(contentsOf: loadURL, configuration: configuration)
        }
    }

    public static func resolvedModelURL(for url: URL) throws -> URL {
        if url.pathExtension != "mlpackage" {
            return url
        }
        // macOS 15 E5 / MPSGraph-backed Core ML packages can retain internal resource
        // references that become invalid after copying a compiled .mlmodelc bundle into a
        // custom cache directory. Compile on demand and use the returned location directly.
        let metricBase = metricBaseName(for: url)
        return try GPTSoVITSRuntimeProfiler.measure("\(metricBase).compile") {
            try MLModel.compileModel(at: url)
        }
    }

    private static func metricBaseName(for url: URL) -> String {
        let stem = url.deletingPathExtension().lastPathComponent
        let sanitized = stem.unicodeScalars.map { scalar -> Character in
            if CharacterSet.alphanumerics.contains(scalar) {
                return Character(scalar)
            }
            return "_"
        }
        return "coreml_loader.\(String(sanitized))"
    }

    private static func compiledCacheURL(for url: URL) throws -> URL {
        try cacheRootDirectory()
            .appendingPathComponent(cacheKey(for: url))
            .appendingPathExtension("mlmodelc")
    }

    private static func cacheRootDirectory() throws -> URL {
        let fileManager = FileManager.default
        if let cacheRoot = fileManager.urls(for: .cachesDirectory, in: .userDomainMask).first {
            return cacheRoot.appendingPathComponent("GPTSoVITSCoreML", isDirectory: true)
        }
        return URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("GPTSoVITSCoreML", isDirectory: true)
    }

    private static func cacheKey(for url: URL) throws -> String {
        let resourceValues = try url.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey])
        let timestamp = resourceValues.contentModificationDate?.timeIntervalSince1970 ?? 0
        let fileSize = resourceValues.fileSize ?? 0
        let payload = "\(url.standardizedFileURL.path)|\(fileSize)|\(timestamp)"
        return String(format: "compiled_%016llx", fnv1a64(payload))
    }

    private static func fnv1a64(_ value: String) -> UInt64 {
        let offset: UInt64 = 0xcbf29ce484222325
        let prime: UInt64 = 0x100000001b3
        var hash = offset
        for byte in value.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* prime
        }
        return hash
    }
}
