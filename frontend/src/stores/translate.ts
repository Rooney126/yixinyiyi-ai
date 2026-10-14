import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { TranslationResult, MemoryRecord } from '../types'

export const useTranslateStore = defineStore('translate', () => {
  const isRecording = ref(false)
  const recordingTime = ref(0)
  const isAnalyzing = ref(false)
  const currentResult = ref<TranslationResult | null>(null)
  const memories = ref<MemoryRecord[]>([])
  const showResult = ref(false)
  const errorMessage = ref('')

  const fakeResults: TranslationResult[] = [
    {
      id: 'demo-1',
      timestamp: Date.now() - 3600000,
      emotionScore: 9,
      emotionState: '开心',
      emotionEmoji: '😊',
      catIntent: '好舒服呀～继续～',
      behaviorAdvice: '猫咪现在心情很好，这是和它互动的好时机',
      translation: '好舒服呀～继续～',
    },
    {
      id: 'demo-2',
      timestamp: Date.now() - 7200000,
      emotionScore: 3,
      emotionState: '防御',
      emotionEmoji: '😰',
      catIntent: '退后！别靠近！',
      behaviorAdvice: '猫咪现在很紧张，给它一点时间和空间，它可以自己调整过来',
      translation: '退后！别靠近！',
    },
    {
      id: 'demo-3',
      timestamp: Date.now() - 10800000,
      emotionScore: 1,
      emotionState: '疼痛',
      emotionEmoji: '😭',
      catIntent: '好痛…别碰这里…',
      behaviorAdvice: '猫咪可能不太舒服，留意它的精神状态和身体情况，持续异常建议联系兽医',
      translation: '好痛…别碰这里…',
    },
  ]

  let timerInterval: ReturnType<typeof setInterval> | null = null

  function startRecording() {
    isRecording.value = true
    recordingTime.value = 0
    timerInterval = setInterval(() => {
      recordingTime.value++
      if (recordingTime.value >= 15) {
        stopRecording()
      }
    }, 1000)
  }

  function stopRecording(blob?: Blob) {
    isRecording.value = false
    if (timerInterval) {
      clearInterval(timerInterval)
      timerInterval = null
    }
    if (blob) {
      uploadAudio(blob)
    }
  }

  async function uploadAudio(blob: Blob) {
    isAnalyzing.value = true
    errorMessage.value = ''
    try {
      const fd = new FormData()
      const fname = blob.type.includes('wav') ? 'recording.wav' : 'recording.webm'
      fd.append('file', blob, fname)
      const resp = await fetch('/api/translate/predict', { method: 'POST', body: fd })
      if (!resp.ok) {
        // 读取服务端具体错误（VAD没切出段/门卫过滤/解码失败等）
        let serverMsg = ''
        try {
          const errData = await resp.json()
          serverMsg = errData?.error || ''
        } catch {}
        const err = new Error(serverMsg || `请求失败(${resp.status})`) as Error & { serverMessage?: string }
        err.serverMessage = serverMsg
        throw err
      }
      const data = await resp.json()
      currentResult.value = {
        id: data.id,
        timestamp: data.timestamp,
        emotionScore: data.emotionScore,
        emotionState: data.emotionState,
        emotionEmoji: data.emotionEmoji,
        catIntent: data.catIntent,
        behaviorAdvice: data.behaviorAdvice,
        confidence: data.confidence,
        confidence_level: data.confidence_level,
        label: data.label,
        translation: data.translation,
        alternative: data.alternative ?? null,
        distribution: data.distribution ?? undefined,
      }
      isAnalyzing.value = false
      showResult.value = true
      addMemory(currentResult.value)
    } catch (e) {
      isAnalyzing.value = false
      // 不再用示例数据冒充结果：显示服务端具体错误或网络错误
      const serverMsg = (e as any)?.serverMessage
      errorMessage.value = serverMsg
        || (e instanceof TypeError ? '网络连接失败，请检查网络' : '请求失败，服务暂时不可用，请稍后重试')
      console.error('翻译请求失败:', e)
    }
  }

  function addMemory(result: TranslationResult) {
    memories.value.unshift({ id: result.id, timestamp: result.timestamp, catName: '', result })
    saveToLocal()
  }

  function dismissResult() {
    showResult.value = false
    // 丢弃暂存音频
    const rid = currentResult.value?.id
    if (rid) {
      fetch(`/api/translate/pending/${rid}`, { method: 'DELETE' }).catch(() => {})
    }
  }

  function loadFromLocal() {
    try {
      const data = localStorage.getItem('translate-memories')
      if (data) memories.value = JSON.parse(data)
    } catch {}
  }

  function saveToLocal() {
    try { localStorage.setItem('translate-memories', JSON.stringify(memories.value)) } catch {}
  }

  function initDemoMemories(catName: string) {
    if (memories.value.length === 0) {
      memories.value = fakeResults.slice(0, 2).map((r) => ({
        id: r.id, timestamp: r.timestamp, catName, result: r,
      }))
      saveToLocal()
    }
  }

  return {
    fakeResults, isRecording, recordingTime, isAnalyzing,
    currentResult, memories, showResult, errorMessage,
    startRecording, stopRecording, uploadAudio, addMemory,
    dismissResult, loadFromLocal, initDemoMemories,
  }
})
