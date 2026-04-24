import CoreML
import Foundation

private enum GPTSoVITSCLIError: Error, CustomStringConvertible {
    case usage(String)
    case missingOption(String)
    case missingValue(String)
    case invalidOption(String)
    case invalidLanguage(String)
    case invalidSplitMethod(String)
    case invalidComputeUnits(String)
    case invalidPromptsJSON(String)
    case conflictingOptions(String)
    case promptTextConflict
    case targetTextConflict
    case promptTextMissing
    case targetTextMissing
    case missingPromptBundle
    case referenceAudioRequired(String)

    var description: String {
        switch self {
        case let .usage(message):
            return message
        case let .missingOption(option):
            return "Missing required option: \(option)"
        case let .missingValue(option):
            return "Missing value for option: \(option)"
        case let .invalidOption(option):
            return "Invalid option: \(option)"
        case let .invalidLanguage(language):
            return "Invalid language: \(language)"
        case let .invalidSplitMethod(splitMethod):
            return "Invalid split method: \(splitMethod)"
        case let .invalidComputeUnits(value):
            return "Invalid compute units: \(value)"
        case let .invalidPromptsJSON(message):
            return message
        case let .conflictingOptions(message):
            return message
        case .promptTextConflict:
            return "Use either --prompt-text or --prompt-text-file, not both."
        case .targetTextConflict:
            return "Use either --target-text or --target-text-file, not both."
        case .promptTextMissing:
            return "Prompt text is required. Use --prompt-text or --prompt-text-file."
        case .targetTextMissing:
            return "Target text is required. Use --target-text or --target-text-file."
        case .missingPromptBundle:
            return "The synthesis bundle does not include prompt_bundle. Use --prompts-json or export a bundle with prompt extraction support."
        case let .referenceAudioRequired(message):
            return message
        }
    }
}

private struct GPTSoVITSCLIOptions {
    let bundleDirectory: String
    let outputWAVPath: String
    let promptText: String
    let targetText: String
    let language: GPTSoVITSTextLanguage?
    let promptLanguage: GPTSoVITSTextLanguage?
    let targetLanguage: GPTSoVITSTextLanguage?
    let splitMethod: GPTSoVITSTextSplitMethod?
    let referenceAudioPath: String?
    let promptReferenceAudioPath: String?
    let conditioningReferenceAudioPath: String?
    let promptsJSONPath: String?
    let noiseScale: Float?
    let seed: UInt64?
    let fragmentIntervalSeconds: Double
    let summaryJSONPath: String?
    let computeUnitsRawValue: String?
}

private enum GPTSoVITSCLIArgumentParser {
    static func parse(arguments: [String]) throws -> GPTSoVITSCLIOptions {
        var bundleDirectory: String?
        var outputWAVPath: String?
        var promptText: String?
        var promptTextFile: String?
        var targetText: String?
        var targetTextFile: String?
        var language: GPTSoVITSTextLanguage?
        var promptLanguage: GPTSoVITSTextLanguage?
        var targetLanguage: GPTSoVITSTextLanguage?
        var splitMethod: GPTSoVITSTextSplitMethod?
        var referenceAudioPath: String?
        var promptReferenceAudioPath: String?
        var conditioningReferenceAudioPath: String?
        var promptsJSONPath: String?
        var noiseScale: Float?
        var seed: UInt64?
        var fragmentIntervalSeconds = 0.3
        var summaryJSONPath: String?
        var computeUnitsRawValue: String?

        var index = 1
        while index < arguments.count {
            let argument = arguments[index]
            switch argument {
            case "-h", "--help":
                throw GPTSoVITSCLIError.usage(helpText())
            case "--bundle-dir":
                bundleDirectory = try value(after: &index, in: arguments, option: argument)
            case "--output-wav":
                outputWAVPath = try value(after: &index, in: arguments, option: argument)
            case "--prompt-text":
                promptText = try value(after: &index, in: arguments, option: argument)
            case "--prompt-text-file":
                promptTextFile = try value(after: &index, in: arguments, option: argument)
            case "--target-text":
                targetText = try value(after: &index, in: arguments, option: argument)
            case "--target-text-file":
                targetTextFile = try value(after: &index, in: arguments, option: argument)
            case "--language":
                let raw = try value(after: &index, in: arguments, option: argument)
                guard let parsed = GPTSoVITSTextLanguage(rawValue: raw) else {
                    throw GPTSoVITSCLIError.invalidLanguage(raw)
                }
                language = parsed
            case "--prompt-language":
                let raw = try value(after: &index, in: arguments, option: argument)
                guard let parsed = GPTSoVITSTextLanguage(rawValue: raw) else {
                    throw GPTSoVITSCLIError.invalidLanguage(raw)
                }
                promptLanguage = parsed
            case "--target-language":
                let raw = try value(after: &index, in: arguments, option: argument)
                guard let parsed = GPTSoVITSTextLanguage(rawValue: raw) else {
                    throw GPTSoVITSCLIError.invalidLanguage(raw)
                }
                targetLanguage = parsed
            case "--split-method":
                let raw = try value(after: &index, in: arguments, option: argument)
                guard let parsed = GPTSoVITSTextSplitMethod(rawValue: raw) else {
                    throw GPTSoVITSCLIError.invalidSplitMethod(raw)
                }
                splitMethod = parsed
            case "--reference-audio":
                referenceAudioPath = try value(after: &index, in: arguments, option: argument)
            case "--prompt-reference-audio":
                promptReferenceAudioPath = try value(after: &index, in: arguments, option: argument)
            case "--conditioning-reference-audio":
                conditioningReferenceAudioPath = try value(after: &index, in: arguments, option: argument)
            case "--prompts-json":
                promptsJSONPath = try value(after: &index, in: arguments, option: argument)
            case "--noise-scale":
                let raw = try value(after: &index, in: arguments, option: argument)
                guard let parsed = Float(raw) else {
                    throw GPTSoVITSCLIError.invalidOption("\(argument)=\(raw)")
                }
                noiseScale = parsed
            case "--seed":
                let raw = try value(after: &index, in: arguments, option: argument)
                guard let parsed = UInt64(raw) else {
                    throw GPTSoVITSCLIError.invalidOption("\(argument)=\(raw)")
                }
                seed = parsed
            case "--fragment-interval":
                let raw = try value(after: &index, in: arguments, option: argument)
                guard let parsed = Double(raw), parsed >= 0 else {
                    throw GPTSoVITSCLIError.invalidOption("\(argument)=\(raw)")
                }
                fragmentIntervalSeconds = parsed
            case "--summary-json":
                summaryJSONPath = try value(after: &index, in: arguments, option: argument)
            case "--compute-units":
                computeUnitsRawValue = try value(after: &index, in: arguments, option: argument)
            default:
                throw GPTSoVITSCLIError.invalidOption(argument)
            }
            index += 1
        }

        guard let resolvedBundleDirectory = bundleDirectory else {
            throw GPTSoVITSCLIError.missingOption("--bundle-dir")
        }
        guard let resolvedOutputWAVPath = outputWAVPath else {
            throw GPTSoVITSCLIError.missingOption("--output-wav")
        }
        if promptText != nil, promptTextFile != nil {
            throw GPTSoVITSCLIError.promptTextConflict
        }
        if targetText != nil, targetTextFile != nil {
            throw GPTSoVITSCLIError.targetTextConflict
        }
        let resolvedPromptText = try resolvedTextValue(
            directValue: promptText,
            filePath: promptTextFile,
            missingError: .promptTextMissing
        )
        let resolvedTargetText = try resolvedTextValue(
            directValue: targetText,
            filePath: targetTextFile,
            missingError: .targetTextMissing
        )
        if language != nil, promptLanguage != nil || targetLanguage != nil {
            throw GPTSoVITSCLIError.conflictingOptions(
                "Use either --language or the explicit --prompt-language/--target-language pair."
            )
        }

        if referenceAudioPath != nil,
           promptReferenceAudioPath != nil || conditioningReferenceAudioPath != nil {
            throw GPTSoVITSCLIError.conflictingOptions(
                "Use either --reference-audio or the explicit --prompt-reference-audio/--conditioning-reference-audio pair."
            )
        }
        if promptsJSONPath != nil, promptReferenceAudioPath != nil {
            throw GPTSoVITSCLIError.conflictingOptions(
                "--prompt-reference-audio cannot be used together with --prompts-json because prompt extraction is bypassed."
            )
        }
        if promptsJSONPath != nil, referenceAudioPath == nil, conditioningReferenceAudioPath == nil {
            throw GPTSoVITSCLIError.referenceAudioRequired(
                "When --prompts-json is provided, you still need conditioning audio via --reference-audio or --conditioning-reference-audio."
            )
        }
        if promptsJSONPath == nil, referenceAudioPath == nil, promptReferenceAudioPath == nil {
            throw GPTSoVITSCLIError.referenceAudioRequired(
                "Reference audio is required. Use --reference-audio or --prompt-reference-audio."
            )
        }

        return GPTSoVITSCLIOptions(
            bundleDirectory: resolvedBundleDirectory,
            outputWAVPath: resolvedOutputWAVPath,
            promptText: resolvedPromptText,
            targetText: resolvedTargetText,
            language: language,
            promptLanguage: promptLanguage,
            targetLanguage: targetLanguage,
            splitMethod: splitMethod,
            referenceAudioPath: referenceAudioPath,
            promptReferenceAudioPath: promptReferenceAudioPath,
            conditioningReferenceAudioPath: conditioningReferenceAudioPath,
            promptsJSONPath: promptsJSONPath,
            noiseScale: noiseScale,
            seed: seed,
            fragmentIntervalSeconds: fragmentIntervalSeconds,
            summaryJSONPath: summaryJSONPath,
            computeUnitsRawValue: computeUnitsRawValue
        )
    }

    static func helpText() -> String {
        """
        Usage:
          gpt-sovits-cli --bundle-dir <bundle-dir> --output-wav <output-wav> \\
            --prompt-text <text>|--prompt-text-file <path> \\
            --target-text <text>|--target-text-file <path> \\
            [--language <zh|all_zh|yue|all_yue|ja|all_ja|ko|all_ko|en>] \\
            [--prompt-language <zh|all_zh|yue|all_yue|ja|all_ja|ko|all_ko|en>] \\
            [--target-language <zh|all_zh|yue|all_yue|ja|all_ja|ko|all_ko|en>] \\
            [--split-method <cut0|cut1|cut2|cut3|cut4|cut5>] \\
            [--reference-audio <wav>] \\
            [--prompt-reference-audio <wav>] [--conditioning-reference-audio <wav>] \\
            [--prompts-json <json>] \\
            [--noise-scale <float>] [--seed <uint64>] \\
            [--fragment-interval <seconds>] \\
            [--compute-units <default|cpu_only|cpu_and_gpu|cpu_and_ne|all>] \\
            [--summary-json <path>]

        Notes:
          - --reference-audio uses the same file for prompt extraction and conditioning.
          - --prompt-reference-audio / --conditioning-reference-audio lets you split prompt extraction and speaker conditioning.
          - --prompts-json bypasses prompt extraction and therefore requires only conditioning audio.
          - This CLI does not fall back to Python text frontends.

        Examples:
          gpt-sovits-cli --bundle-dir artifacts/coreml/chinese_synthesis_bundle \\
            --output-wav out.wav \\
            --prompt-text-file test.lab \\
            --target-text "你好，这是 Swift CLI 直出的音频。" \\
            --reference-audio test.wav \\
            --compute-units cpu_and_ne

          gpt-sovits-cli --bundle-dir artifacts/coreml/chinese_synthesis_bundle \\
            --output-wav out.wav \\
            --prompt-text-file test.lab \\
            --target-text "Hello from the native CLI." \\
            --prompt-language zh \\
            --target-language en \\
            --reference-audio test.wav \\
            --compute-units cpu_and_ne
        """
    }

    private static func value(
        after index: inout Int,
        in arguments: [String],
        option: String
    ) throws -> String {
        let nextIndex = index + 1
        guard nextIndex < arguments.count else {
            throw GPTSoVITSCLIError.missingValue(option)
        }
        index = nextIndex
        return arguments[nextIndex]
    }

    private static func resolvedTextValue(
        directValue: String?,
        filePath: String?,
        missingError: GPTSoVITSCLIError
    ) throws -> String {
        if let directValue {
            return directValue
        }
        if let filePath {
            let data = try Data(contentsOf: URL(fileURLWithPath: filePath, isDirectory: false))
            return String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        throw missingError
    }
}

private func resolvedModelConfiguration(rawValue: String?) throws -> (configuration: MLModelConfiguration, label: String) {
    let configuration = MLModelConfiguration()
    let raw = (rawValue ?? ProcessInfo.processInfo.environment["GPTSOVITS_COREML_COMPUTE_UNITS"] ?? "default").lowercased()
    switch raw {
    case "default":
        return (configuration, "default")
    case "cpu_only":
        configuration.computeUnits = .cpuOnly
        return (configuration, "cpu_only")
    case "cpu_and_gpu":
        configuration.computeUnits = .cpuAndGPU
        return (configuration, "cpu_and_gpu")
    case "cpu_and_ne":
        configuration.computeUnits = .cpuAndNeuralEngine
        return (configuration, "cpu_and_ne")
    case "all":
        configuration.computeUnits = .all
        return (configuration, "all")
    default:
        throw GPTSoVITSCLIError.invalidComputeUnits(raw)
    }
}

private func loadPromptTokens(from path: String) throws -> [Int32] {
    let url = URL(fileURLWithPath: path, isDirectory: false)
    let data = try Data(contentsOf: url)
    let object = try JSONSerialization.jsonObject(with: data)
    let values: [Any]
    if let items = object as? [Any] {
        values = items
    } else if let dictionary = object as? [String: Any],
              let items = dictionary["prompts"] as? [Any] {
        values = items
    } else {
        throw GPTSoVITSCLIError.invalidPromptsJSON(
            "Invalid prompts JSON at \(path). Expected an array or an object with a 'prompts' array."
        )
    }

    let prompts = values.compactMap { item -> Int32? in
        if let number = item as? NSNumber {
            return number.int32Value
        }
        if let string = item as? String, let value = Int32(string) {
            return value
        }
        return nil
    }
    guard prompts.count == values.count else {
        throw GPTSoVITSCLIError.invalidPromptsJSON(
            "Invalid prompts JSON at \(path). All prompt entries must be numeric."
        )
    }
    return prompts
}

private func writeJSONPayload(_ payload: [String: Any], to path: String) throws {
    let url = URL(fileURLWithPath: path, isDirectory: false)
    try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(),
        withIntermediateDirectories: true,
        attributes: nil
    )
    let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: url)
}

private func formatShape(_ array: MLMultiArray) -> [Int] {
    array.shape.map { Int(truncating: $0) }
}

private func int32Values(_ array: MLMultiArray) -> [Int32] {
    (0..<array.count).map { Int32(truncating: array[$0]) }
}

private func summarizeAudioExport(_ export: GPTSoVITSChineseSynthesisAudioExport) -> [String: Any] {
    [
        "output_sample_rate": Int(export.outputSampleRate),
        "hop_length": export.hopLength,
        "gap_sample_count": export.gapSampleCount,
        "sample_count": export.samples.count,
        "segment_active_frame_counts": export.segmentExports.map(\.activeFrameCount),
        "segment_active_sample_counts": export.segmentExports.map(\.activeSampleCount),
        "segment_total_sample_counts": export.segmentExports.map(\.totalSampleCount),
        "segment_trailing_padding_sample_counts": export.segmentExports.map(\.trailingPaddingSampleCount),
        "segment_sample_strides": export.segmentExports.map(\.sampleStride),
    ]
}

@main
struct GPTSoVITSCLI {
    static func main() throws {
        do {
            let options = try GPTSoVITSCLIArgumentParser.parse(arguments: CommandLine.arguments)
            let configuration = try resolvedModelConfiguration(rawValue: options.computeUnitsRawValue)
            let bundleURL = URL(fileURLWithPath: options.bundleDirectory, isDirectory: true)
            let outputWAVURL = URL(fileURLWithPath: options.outputWAVPath, isDirectory: false)

            let driver = try GPTSoVITSChineseSynthesisBundleDriver(
                bundleDirectory: bundleURL,
                configuration: configuration.configuration
            )
            let defaultLanguage = GPTSoVITSTextLanguage(rawValue: driver.manifest.runtime.defaults.language) ?? .zh
            let defaultSplitMethod = GPTSoVITSTextSplitMethod(rawValue: driver.manifest.runtime.defaults.splitMethod) ?? .cut5
            let resolvedPromptLanguage = options.promptLanguage ?? options.language ?? defaultLanguage
            let resolvedTargetLanguage = options.targetLanguage ?? options.language ?? defaultLanguage
            let resolvedSplitMethod = options.splitMethod ?? defaultSplitMethod
            let isCrossLingualMode = resolvedPromptLanguage != resolvedTargetLanguage

            let prompts = try options.promptsJSONPath.map(loadPromptTokens(from:))
            if prompts == nil, driver.manifest.artifacts.promptBundle == nil {
                throw GPTSoVITSCLIError.missingPromptBundle
            }

            let result: GPTSoVITSChineseSynthesisResult
            let referenceMode: String
            if let prompts {
                let conditioningAudioPath = options.referenceAudioPath ?? options.conditioningReferenceAudioPath!
                let conditioningAudio = try ReferenceAudioSamples(
                    contentsOf: URL(fileURLWithPath: conditioningAudioPath, isDirectory: false)
                )
                if isCrossLingualMode {
                    result = try driver.synthesizeCrossLingual(
                        prompts: prompts,
                        promptText: options.promptText,
                        targetText: options.targetText,
                        promptLanguage: resolvedPromptLanguage,
                        targetLanguage: resolvedTargetLanguage,
                        referenceAudio: conditioningAudio,
                        splitMethod: resolvedSplitMethod,
                        noiseScale: options.noiseScale,
                        seed: options.seed
                    )
                } else {
                    result = try driver.synthesize(
                        prompts: prompts,
                        promptText: options.promptText,
                        targetText: options.targetText,
                        referenceAudio: conditioningAudio,
                        language: resolvedTargetLanguage,
                        splitMethod: resolvedSplitMethod,
                        noiseScale: options.noiseScale,
                        seed: options.seed
                    )
                }
                referenceMode = "external_prompts_json"
            } else if let sharedReferenceAudioPath = options.referenceAudioPath {
                let referenceAudio = try ReferenceAudioSamples(
                    contentsOf: URL(fileURLWithPath: sharedReferenceAudioPath, isDirectory: false)
                )
                if isCrossLingualMode {
                    result = try driver.synthesizeCrossLingual(
                        promptText: options.promptText,
                        targetText: options.targetText,
                        promptLanguage: resolvedPromptLanguage,
                        targetLanguage: resolvedTargetLanguage,
                        referenceAudio: referenceAudio,
                        splitMethod: resolvedSplitMethod,
                        noiseScale: options.noiseScale,
                        seed: options.seed
                    )
                } else {
                    result = try driver.synthesize(
                        promptText: options.promptText,
                        targetText: options.targetText,
                        referenceAudio: referenceAudio,
                        language: resolvedTargetLanguage,
                        splitMethod: resolvedSplitMethod,
                        noiseScale: options.noiseScale,
                        seed: options.seed
                    )
                }
                referenceMode = "shared_reference_audio"
            } else {
                let promptReferenceAudioPath = options.promptReferenceAudioPath!
                let conditioningReferenceAudioPath = options.conditioningReferenceAudioPath ?? promptReferenceAudioPath
                let promptReferenceAudio = try ReferenceAudioSamples(
                    contentsOf: URL(fileURLWithPath: promptReferenceAudioPath, isDirectory: false)
                )
                let conditioningReferenceAudio = try ReferenceAudioSamples(
                    contentsOf: URL(fileURLWithPath: conditioningReferenceAudioPath, isDirectory: false)
                )
                if isCrossLingualMode {
                    result = try driver.synthesizeCrossLingual(
                        promptText: options.promptText,
                        targetText: options.targetText,
                        promptLanguage: resolvedPromptLanguage,
                        targetLanguage: resolvedTargetLanguage,
                        promptReferenceAudio: promptReferenceAudio,
                        conditioningReferenceAudio: conditioningReferenceAudio,
                        splitMethod: resolvedSplitMethod,
                        noiseScale: options.noiseScale,
                        seed: options.seed
                    )
                } else {
                    result = try driver.synthesize(
                        promptText: options.promptText,
                        targetText: options.targetText,
                        promptReferenceAudio: promptReferenceAudio,
                        conditioningReferenceAudio: conditioningReferenceAudio,
                        language: resolvedTargetLanguage,
                        splitMethod: resolvedSplitMethod,
                        noiseScale: options.noiseScale,
                        seed: options.seed
                    )
                }
                referenceMode = promptReferenceAudioPath == conditioningReferenceAudioPath
                    ? "shared_explicit_reference_audio"
                    : "split_reference_audio"
            }

            let audioExport = GPTSoVITSChineseSynthesisAudioExporter.export(
                from: result,
                pipeline: driver.pipeline,
                fragmentIntervalSeconds: options.fragmentIntervalSeconds
            )
            try GPTSoVITSChineseSynthesisAudioExporter.writeWAV(
                samples: audioExport.samples,
                sampleRate: audioExport.outputSampleRate,
                to: outputWAVURL
            )

            let payload: [String: Any] = [
                "bundle_dir": bundleURL.path,
                "output_wav": outputWAVURL.path,
                "coreml_compute_units": configuration.label,
                "reference_mode": referenceMode,
                "prompt_text": options.promptText,
                "target_text": options.targetText,
                "language": resolvedTargetLanguage.rawValue,
                "prompt_language": resolvedPromptLanguage.rawValue,
                "target_language": resolvedTargetLanguage.rawValue,
                "language_mode": isCrossLingualMode ? "cross_lingual" : "same_language",
                "split_method": resolvedSplitMethod.rawValue,
                "prompt_source": prompts == nil ? "bundle_extract" : "external_prompts_json",
                "prompt_count": result.prompts.count,
                "prompt_preview": Array(result.prompts.prefix(32)),
                "prompt_tail": Array(result.prompts.suffix(16)),
                "prompt_extraction_prompt_count": result.promptExtraction?.promptCount as Any,
                "prompt_extraction_ssl_shape": result.promptExtraction.map { formatShape($0.sslContent) } as Any,
                "prompt_extraction_prompt_shape": result.promptExtraction.map { formatShape($0.promptSemantic) } as Any,
                "prepared_prompt_normalized_text": result.prompt.normalizedText,
                "prepared_prompt_phone_count": result.prompt.phoneCount,
                "prepared_prompt_backend": result.prompt.textFrontendBackend,
                "segment_count": result.segments.count,
                "segment_backends": result.segments.map { $0.preparedSegment.input.textFrontendBackend },
                "segment_normalized_texts": result.segments.map { $0.preparedSegment.input.normalizedText },
                "segment_phone_counts": result.segments.map { $0.preparedSegment.input.phoneCount },
                "segment_semantic_code_counts": result.segments.map { $0.synthesis.semanticCodes.count },
                "segment_cache_lens": result.segments.map { int32Values($0.synthesis.t2sState.cacheLen) },
                "segment_next_positions": result.segments.map { int32Values($0.synthesis.t2sState.nextPosition) },
                "segment_audio_shapes": result.segments.map { formatShape($0.synthesis.vitsResult.audio) },
                "audio_export": summarizeAudioExport(audioExport),
            ]

            if let summaryJSONPath = options.summaryJSONPath {
                try writeJSONPayload(payload, to: summaryJSONPath)
            }
            let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
            print(String(decoding: data, as: UTF8.self))
        } catch let error as GPTSoVITSCLIError {
            fputs("\(error.description)\n", stderr)
            if case .usage = error {
                exit(0)
            }
            exit(1)
        }
    }
}
