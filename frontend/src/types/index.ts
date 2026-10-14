export interface TranslationResult {
  id: string
  timestamp: number
  emotionScore: number
  emotionState: string
  emotionEmoji: string
  catIntent: string
  behaviorAdvice: string
  confidence?: number
  confidence_level?: 'high' | 'medium' | 'low'
  label?: string
  translation?: string
  alternative?: {
    label: string
    translation: string
    emoji: string
    state: string
    confidence: number
  } | null
  distribution?: Array<{
    label: string
    prob: number
    emoji: string
    translation: string
    state: string
  }>
  audioBlob?: Blob
  audioUrl?: string
}

export interface MemoryRecord {
  id: string
  timestamp: number
  catName: string
  result: TranslationResult
}

export interface UserProfile {
  catName: string
  avatar: string
  badges: string[]
}

export type HomeChannel = 'translate' | 'medical' | 'market' | 'community' | 'games'
