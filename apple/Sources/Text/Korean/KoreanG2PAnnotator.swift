import Foundation

public enum KoreanG2PAnnotatorError: LocalizedError {
    case tokenAlignmentMismatch(String)

    public var errorDescription: String? {
        switch self {
        case let .tokenAlignmentMismatch(text):
            return "韩语 annotate 无法对齐原文与 mecab token: \(text)"
        }
    }
}

public final class KoreanG2PAnnotator {
    private let tagger: KoreanMecabTagger

    public init(tagger: KoreanMecabTagger) {
        self.tagger = tagger
    }

    public func annotate(_ text: String) throws -> String {
        let tokens = try tagger.pos(text)
        let stripped = text.filter { !$0.isWhitespace }
        guard stripped == tokens.map(\.surface).joined() else {
            return text
        }

        let blankOffsets = text.enumerated().compactMap { offset, character in
            character == " " ? offset : nil
        }

        var tagSequence = ""
        for token in tokens {
            let terminalTag = token.tag.split(separator: "+").last.map(String.init) ?? token.tag
            let mappedTag: Character = terminalTag == "NNBC"
                ? "B"
                : terminalTag.first ?? "_"
            if token.surface.count > 1 {
                tagSequence += String(repeating: "_", count: token.surface.count - 1)
            }
            tagSequence.append(mappedTag)
        }

        for offset in blankOffsets.sorted() where offset <= tagSequence.count {
            let index = tagSequence.index(tagSequence.startIndex, offsetBy: offset)
            tagSequence.insert(" ", at: index)
        }

        var annotated = ""
        let tags = Array(tagSequence)
        for (index, character) in text.enumerated() {
            guard index < tags.count else { break }
            let tag = tags[index]
            annotated.append(character)

            if character == "의", tag == "J" {
                annotated += "/J"
                continue
            }

            guard let jongseong = Self.trailingJongseong(of: character) else {
                if tag == "B" {
                    annotated += "/B"
                }
                continue
            }

            if tag == "E", jongseong == "ᆯ" {
                annotated += "/E"
            } else if tag == "V", Self.verbNieunFinals.contains(jongseong) {
                annotated += "/P"
            } else if tag == "B" {
                annotated += "/B"
            }
        }
        return annotated
    }

    private static let verbNieunFinals: Set<Character> = ["ᆫ", "ᆬ", "ᆷ", "ᆱ", "ᆰ", "ᆲ", "ᆴ"]

    private static func trailingJongseong(of character: Character) -> Character? {
        guard let scalar = character.unicodeScalars.first,
              scalar.value >= 0xAC00, scalar.value <= 0xD7A3 else {
            return nil
        }
        let syllableIndex = Int(scalar.value - 0xAC00)
        let jongIndex = syllableIndex % 28
        guard jongIndex > 0,
              let jongScalar = UnicodeScalar(0x11A7 + jongIndex) else {
            return nil
        }
        return Character(jongScalar)
    }
}
