import Foundation

public struct GPTSoVITSRuntimeProfileMetric {
    public let name: String
    public let count: Int
    public let totalSeconds: Double
    public let averageSeconds: Double
    public let minSeconds: Double
    public let maxSeconds: Double
}

private struct GPTSoVITSRuntimeProfileAccumulator {
    var count: Int = 0
    var totalSeconds: Double = 0
    var minSeconds: Double = .greatestFiniteMagnitude
    var maxSeconds: Double = 0

    mutating func record(_ seconds: Double) {
        count += 1
        totalSeconds += seconds
        minSeconds = min(minSeconds, seconds)
        maxSeconds = max(maxSeconds, seconds)
    }

    func snapshot(name: String) -> GPTSoVITSRuntimeProfileMetric {
        let average = count > 0 ? totalSeconds / Double(count) : 0
        let minValue = count > 0 ? minSeconds : 0
        return GPTSoVITSRuntimeProfileMetric(
            name: name,
            count: count,
            totalSeconds: totalSeconds,
            averageSeconds: average,
            minSeconds: minValue,
            maxSeconds: maxSeconds
        )
    }
}

public enum GPTSoVITSRuntimeProfiler {
    private static let lock = NSLock()
    private static var enabled = false
    private static var accumulators: [String: GPTSoVITSRuntimeProfileAccumulator] = [:]

    public static func setEnabled(_ newValue: Bool) {
        lock.lock()
        enabled = newValue
        lock.unlock()
    }

    public static var isEnabled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return enabled
    }

    public static func reset() {
        lock.lock()
        accumulators.removeAll()
        lock.unlock()
    }

    @discardableResult
    public static func measure<T>(_ name: String, _ body: () throws -> T) rethrows -> T {
        guard isEnabled else {
            return try body()
        }
        let start = CFAbsoluteTimeGetCurrent()
        do {
            let value = try body()
            record(name: name, seconds: CFAbsoluteTimeGetCurrent() - start)
            return value
        } catch {
            record(name: name, seconds: CFAbsoluteTimeGetCurrent() - start)
            throw error
        }
    }

    public static func snapshot() -> [GPTSoVITSRuntimeProfileMetric] {
        lock.lock()
        let metrics = accumulators
            .map { $0.value.snapshot(name: $0.key) }
            .sorted {
                if $0.totalSeconds == $1.totalSeconds {
                    return $0.name < $1.name
                }
                return $0.totalSeconds > $1.totalSeconds
            }
        lock.unlock()
        return metrics
    }

    private static func record(name: String, seconds: Double) {
        lock.lock()
        var accumulator = accumulators[name, default: GPTSoVITSRuntimeProfileAccumulator()]
        accumulator.record(seconds)
        accumulators[name] = accumulator
        lock.unlock()
    }
}
