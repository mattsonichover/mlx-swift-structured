//
//  GrammarMaskedLogitProcessor.swift
//  MLXStructured
//
//  Created by Ivan Petrukha on 14.09.2025.
//

import MLXLMCommon
import MLX

public final class GrammarMaskedLogitProcessor: LogitProcessor, @unchecked Sendable {
    
    public let grammarMatcher: GrammarMatcher
    
    public init(grammarMatcher: GrammarMatcher) {
        self.grammarMatcher = grammarMatcher
    }
    
    public func prompt(_ prompt: MLXArray) {
        grammarMatcher.reset()
    }
    
    public func process(logits: MLXArray) -> MLXArray {
      let mask = grammarMatcher.nextTokenMask()
      let logitsVocab = logits.shape[logits.ndim - 1]
      let maskElements = mask.shape.reduce(1, *)
      if maskElements == logitsVocab {
          return logits + mask
      }
      // Model output dim differs from tokenizer vocab (common in multimodal models).
      // Pad with -inf to block reserved/unknown token slots.
      let padding = MLXArray(Array(repeating: -Float.infinity, count: logitsVocab - maskElements))
      let paddedMask = MLX.concatenated([mask.reshaped([-1]), padding])
      return logits + paddedMask
  }
    
    public func didSample(token: MLXArray) {
        grammarMatcher.advance(token: token)
    }
}
