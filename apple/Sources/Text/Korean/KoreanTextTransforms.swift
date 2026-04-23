import Foundation

public final class KoreanTextTransforms {
    private let projectAssets: KoreanFrontendRuntimeAssets.ProjectAssets
    private let numeralsAssets: KoreanFrontendRuntimeAssets.G2pk2NumeralsAssets
    private let koreanClassifierSet: Set<String>

    public init(bundle: KoreanFrontendBundle) {
        self.projectAssets = bundle.runtimeAssets.projectAssets
        self.numeralsAssets = bundle.runtimeAssets.g2pk2NumeralsAssets
        self.koreanClassifierSet = Set(bundle.runtimeAssets.g2pk2NumeralsAssets.boundNouns)
    }

    public func latinToHangul(_ text: String) -> String {
        applyRegexReplacements(
            projectAssets.latinToHangul,
            to: text,
            options: [.caseInsensitive]
        )
    }

    public func numberToHangul(_ text: String) -> String {
        var output = text
        let pattern = #"(\d[\d,]*)([\uac00-\ud71f]+)"#
        let regex = try? NSRegularExpression(pattern: pattern)
        let matches = regex?.matches(in: output, range: NSRange(output.startIndex..., in: output)) ?? []
        var seen = Set<String>()
        let tokens = matches.compactMap { match -> (String, String)? in
            guard match.numberOfRanges == 3,
                  let numberRange = Range(match.range(at: 1), in: output),
                  let classifierRange = Range(match.range(at: 2), in: output) else {
                return nil
            }
            let number = String(output[numberRange])
            let classifier = String(output[classifierRange])
            let key = "\(number)\u{0000}\(classifier)"
            guard seen.insert(key).inserted else {
                return nil
            }
            return (number, classifier)
        }

        for (number, classifier) in tokens {
            let sino = !usesPureKoreanClassifier(classifier)
            let spoken = processNumber(number, sino: sino)
            output = output.replacingOccurrences(of: "\(number)\(classifier)", with: "\(spoken)\(classifier)")
        }

        for (digit, spoken) in zip(numeralsAssets.digits, numeralsAssets.digitNames) {
            output = output.replacingOccurrences(of: digit, with: spoken)
        }
        return output
    }

    public func divideHangul(_ text: String) -> String {
        var output = String()
        output.reserveCapacity(text.count * 2)
        for character in text {
            output += Self.compatibilityString(for: character)
        }
        return applyRegexReplacements(projectAssets.hangulDivided, to: output)
    }

    public func fixG2PK2Error(_ text: String) -> String {
        let characters = Array(text)
        guard characters.count >= 5 else {
            return text
        }

        var output = ""
        var index = 0
        while index < characters.count - 4 {
            let tri = String(characters[index..<(index + 3)])
            if (tri == "ㅇㅡㄹ" || tri == "ㄹㅡㄹ"),
               characters[index + 3] == " ",
               characters[index + 4] == "ㄹ" {
                output += tri
                output += " "
                output += "ㄴ"
                index += 5
            } else {
                output.append(characters[index])
                index += 1
            }
        }
        if index < characters.count {
            output += String(characters[index...])
        }
        return output
    }

    private func usesPureKoreanClassifier(_ classifier: String) -> Bool {
        let firstTwo = String(classifier.prefix(2))
        let firstOne = String(classifier.prefix(1))
        return koreanClassifierSet.contains(firstTwo) || koreanClassifierSet.contains(firstOne)
    }

    private func processNumber(_ number: String, sino: Bool) -> String {
        let compact = number.replacingOccurrences(of: ",", with: "")
        if compact == "0" {
            return "영"
        }
        if !sino, compact == "20" {
            return "스무"
        }

        let digitToName = Dictionary(uniqueKeysWithValues: zip(numeralsAssets.nonZeroDigits, numeralsAssets.nonZeroDigitNames))
        let digitToModifier = Dictionary(uniqueKeysWithValues: zip(numeralsAssets.nonZeroDigits, numeralsAssets.modifiers))
        let digitToDecimal = Dictionary(uniqueKeysWithValues: zip(numeralsAssets.nonZeroDigits, numeralsAssets.decimals))

        var spoken = [String]()
        let digits = Array(compact)
        for (offset, digitCharacter) in digits.enumerated() {
            let reverseIndex = digits.count - offset - 1
            let digit = String(digitCharacter)
            var name = ""

            if sino {
                if reverseIndex == 0 {
                    name = digitToName[digit] ?? ""
                } else if reverseIndex == 1 {
                    name = (digitToName[digit] ?? "") + "십"
                    name = name.replacingOccurrences(of: "일십", with: "십")
                }
            } else {
                if reverseIndex == 0 {
                    name = digitToModifier[digit] ?? ""
                } else if reverseIndex == 1 {
                    name = digitToDecimal[digit] ?? ""
                }
            }

            if digit == "0" {
                if reverseIndex % 4 == 0 {
                    let suffix = Array(spoken.suffix(min(3, spoken.count))).joined()
                    if suffix.isEmpty {
                        spoken.append("")
                        continue
                    }
                } else {
                    spoken.append("")
                    continue
                }
            }

            switch reverseIndex {
            case 2:
                name = (digitToName[digit] ?? "") + "백"
                name = name.replacingOccurrences(of: "일백", with: "백")
            case 3:
                name = (digitToName[digit] ?? "") + "천"
                name = name.replacingOccurrences(of: "일천", with: "천")
            case 4:
                name = (digitToName[digit] ?? "") + "만"
                name = name.replacingOccurrences(of: "일만", with: "만")
            case 5:
                name = (digitToName[digit] ?? "") + "십"
                name = name.replacingOccurrences(of: "일십", with: "십")
            case 6:
                name = (digitToName[digit] ?? "") + "백"
                name = name.replacingOccurrences(of: "일백", with: "백")
            case 7:
                name = (digitToName[digit] ?? "") + "천"
                name = name.replacingOccurrences(of: "일천", with: "천")
            case 8:
                name = (digitToName[digit] ?? "") + "억"
            case 9:
                name = (digitToName[digit] ?? "") + "십"
            case 10:
                name = (digitToName[digit] ?? "") + "백"
            case 11:
                name = (digitToName[digit] ?? "") + "천"
            case 12:
                name = (digitToName[digit] ?? "") + "조"
            case 13:
                name = (digitToName[digit] ?? "") + "십"
            case 14:
                name = (digitToName[digit] ?? "") + "백"
            case 15:
                name = (digitToName[digit] ?? "") + "천"
            default:
                break
            }
            spoken.append(name)
        }
        return spoken.joined()
    }

    private func applyRegexReplacements(
        _ replacements: [KoreanFrontendRuntimeAssets.RegexReplacement],
        to text: String,
        options: NSRegularExpression.Options = []
    ) -> String {
        var output = text
        for replacement in replacements {
            guard let regex = try? NSRegularExpression(pattern: replacement.pattern, options: options) else {
                continue
            }
            output = regex.stringByReplacingMatches(
                in: output,
                options: [],
                range: NSRange(output.startIndex..., in: output),
                withTemplate: replacement.replacement
            )
        }
        return output
    }

    private static func compatibilityString(for character: Character) -> String {
        if character.unicodeScalars.count > 1 {
            return character.unicodeScalars.map(Self.compatibilityString(for:)).joined()
        }
        guard let scalar = character.unicodeScalars.first else {
            return String(character)
        }
        return compatibilityString(for: scalar)
    }

    private static func compatibilityString(for scalar: UnicodeScalar) -> String {
        if scalar.value >= 0x1100, scalar.value <= 0x1112 {
            return compatibilityChoseong[Int(scalar.value - 0x1100)]
        }
        if scalar.value >= 0x1161, scalar.value <= 0x1175 {
            return compatibilityJungseong[Int(scalar.value - 0x1161)]
        }
        if scalar.value >= 0x11A8, scalar.value <= 0x11C2 {
            return compatibilityJongseong[Int(scalar.value - 0x11A7)]
        }
        guard scalar.value >= 0xAC00, scalar.value <= 0xD7A3 else {
            return String(scalar)
        }

        let syllableIndex = Int(scalar.value - 0xAC00)
        let choseongIndex = syllableIndex / 588
        let jungseongIndex = (syllableIndex % 588) / 28
        let jongseongIndex = syllableIndex % 28

        var output = compatibilityChoseong[choseongIndex] + compatibilityJungseong[jungseongIndex]
        if jongseongIndex > 0 {
            output += compatibilityJongseong[jongseongIndex]
        }
        return output
    }

    private static let compatibilityChoseong = [
        "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
        "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
    ]

    private static let compatibilityJungseong = [
        "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ",
        "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
    ]

    private static let compatibilityJongseong = [
        "",
        "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ",
        "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ",
        "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
    ]
}
