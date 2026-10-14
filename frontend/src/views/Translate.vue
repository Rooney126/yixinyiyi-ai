<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useTranslateStore } from '../stores/translate'
import { useProfileStore } from '../stores/profile'
import ResultCard from '../components/ResultCard.vue'
import { Mic, Square, Upload, Search, Music } from 'lucide-vue-next'

const translate = useTranslateStore()
const profile = useProfileStore()
const fileInput = ref<HTMLInputElement | null>(null)

// 录音状态
let mediaRecorder: MediaRecorder | null = null
let audioChunks: Blob[] = []
// WebAudio PCM 录音状态（兼容无 MediaRecorder 的浏览器，如华为/QQ 浏览器）
let waCtx: AudioContext | null = null
let waSource: MediaStreamAudioSourceNode | null = null
let waProcessor: ScriptProcessorNode | null = null
let waStream: MediaStream | null = null
let waChunks: Float32Array[] = []

function handleRecordClick() {
  if (translate.isRecording) {
    // 停止录音 → 发送
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop()
    } else {
      stopWebAudioRecording()
    }
  } else {
    startMediaRecorder()
  }
}

// ─── WebAudio PCM 录音（输出 WAV，不依赖 MediaRecorder）───
function startWebAudioRecording(stream: MediaStream) {
  waStream = stream
  waChunks = []
  const Ctx = window.AudioContext || (window as any).webkitAudioContext
  waCtx = new Ctx()
  waSource = waCtx.createMediaStreamSource(stream)
  waProcessor = waCtx.createScriptProcessor(4096, 1, 1)
  waProcessor.onaudioprocess = (e) => {
    waChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)))
  }
  waSource.connect(waProcessor)
  waProcessor.connect(waCtx.destination)  // 静音输出，触发处理回调
  translate.startRecording()
}

function stopWebAudioRecording() {
  if (!waCtx || !waSource || !waProcessor) return
  waSource.disconnect()
  waProcessor.disconnect()
  waStream?.getTracks().forEach(t => t.stop())
  const sampleRate = waCtx.sampleRate
  const blob = encodeWav(waChunks, sampleRate)
  waCtx.close().catch(() => {})
  waCtx = null; waSource = null; waProcessor = null; waStream = null; waChunks = []
  translate.stopRecording(blob)
}

function encodeWav(chunks: Float32Array[], sampleRate: number): Blob {
  const total = chunks.reduce((s, c) => s + c.length, 0)
  const buf = new ArrayBuffer(44 + total * 2)
  const v = new DataView(buf)
  const writeStr = (off: number, s: string) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)) }
  writeStr(0, 'RIFF'); v.setUint32(4, 36 + total * 2, true); writeStr(8, 'WAVE')
  writeStr(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true)
  v.setUint16(22, 1, true); v.setUint32(24, sampleRate, true)
  v.setUint32(28, sampleRate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true)
  writeStr(36, 'data'); v.setUint32(40, total * 2, true)
  let off = 44
  for (const c of chunks) {
    for (let i = 0; i < c.length; i++) {
      const s = Math.max(-1, Math.min(1, c[i]))
      v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true)
      off += 2
    }
  }
  return new Blob([buf], { type: 'audio/wav' })
}

async function startMediaRecorder() {
  if (!navigator.mediaDevices?.getUserMedia) {
    return systemFallback('浏览器不支持录音')
  }
  let stream: MediaStream | null = null
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  } catch (e) {
    return systemFallback('麦克风授权失败')
  }
  // ① 优先 MediaRecorder（标准浏览器，输出 webm/mp4）
  if (typeof MediaRecorder !== 'undefined') {
    try {
      // 选择浏览器支持的录音格式；都不支持则不指定 mimeType 用默认
      let mime = ''
      if (typeof MediaRecorder.isTypeSupported === 'function') {
        if (MediaRecorder.isTypeSupported('audio/webm')) mime = 'audio/webm'
        else if (MediaRecorder.isTypeSupported('audio/mp4')) mime = 'audio/mp4'
        else if (MediaRecorder.isTypeSupported('audio/ogg')) mime = 'audio/ogg'
      }
      mediaRecorder = mime
        ? new MediaRecorder(stream, { mimeType: mime })
        : new MediaRecorder(stream)
      audioChunks = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunks.push(e.data)
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunks, { type: mediaRecorder?.mimeType || mime || 'audio/webm' })
        stream?.getTracks().forEach(t => t.stop())
        translate.stopRecording(blob)
      }

      translate.startRecording()
      mediaRecorder.start()
      return  // 在线录音成功，退出
    } catch (e) {
      console.log('MediaRecorder 不可用', e)
    }
  }
  // ② MediaRecorder 缺失/失败 → WebAudio PCM 录音（复用同一个 stream）
  if (typeof AudioContext !== 'undefined' && stream) {
    try {
      startWebAudioRecording(stream)
      return
    } catch (e) {
      console.log('WebAudio 录音失败', e)
    }
  }
  // ③ 都失败 → 释放流，打开系统录音机
  stream?.getTracks().forEach(t => t.stop())
  systemFallback('网页录音失败')
}

function systemFallback(reason: string) {
  try {
    const HINT_KEY = 'catmood-record-hint2'
    if (!localStorage.getItem(HINT_KEY)) {
      localStorage.setItem(HINT_KEY, '1')
      alert(`网页录音失败（${reason}），将打开系统录音机，录完会自动上传识别。`)
    }
  } catch {}
  const input = document.createElement('input')
  input.type = 'file'
  input.accept = 'audio/*'
  input.setAttribute('capture', 'environment')
  input.onchange = (e: any) => {
    if (e.target?.files?.[0]) {
      translate.uploadAudio(e.target.files[0])
    }
  }
  input.click()
}

function handleFileSelect(e: Event) {
  const input = e.target as HTMLInputElement
  if (input.files?.[0]) {
    translate.uploadAudio(input.files[0])
  }
}

const formatTime = (s: number) => `${s}s`

onMounted(() => { translate.loadFromLocal() })
</script>

<template>
  <div class="flex flex-col items-center min-h-full pt-8 pb-4 relative">
    <div class="absolute top-0 left-0 right-0 h-64 bg-gradient-to-b from-[#F5E6D3]/40 to-transparent pointer-events-none"></div>

    <div class="relative z-10 w-full px-6 flex flex-col items-center">
      <!-- 标题 -->
      <div class="text-center mb-10">
        <p class="text-sm text-[#8B9D8B] font-light tracking-widest">猫咪翻译</p>
        <p v-if="!profile.hasCat" class="text-xs text-[#B5D8CC] mt-1 font-light">在「我的」为猫咪取名</p>
        <p v-else class="text-xs text-[#B5D8CC] mt-1 font-light">{{ profile.catName }} 在说什么呢？</p>
      </div>

      <!-- 分析中 -->
      <div v-if="translate.isAnalyzing" class="flex flex-col items-center mb-10">
        <div class="relative w-32 h-32 mb-6">
          <div class="absolute inset-0 bg-[#F5E6D3]/30 rounded-full animate-ping"></div>
          <div class="relative w-32 h-32 rounded-full bg-white/80 backdrop-blur border border-[#D4C4B0] flex items-center justify-center">
            <Search :size="36" class="text-[#8B9D8B]" :stroke-width="1.5" />
          </div>
        </div>
        <p class="text-base text-[#5C4E3E] font-light">正在分析猫叫声…</p>
        <p class="text-xs text-[#8B9D8B] mt-1 font-light">稍等一下，马上出结果</p>
      </div>

      <!-- 录音区 -->
      <div v-else class="flex flex-col items-center mb-10">
        <div class="relative mb-6">
          <div v-if="translate.isRecording" class="absolute inset-0">
            <div class="absolute inset-0 rounded-full bg-red-100/30 animate-ping"></div>
            <div class="absolute inset-2 rounded-full bg-red-50/30 animate-ping" style="animation-delay:0.3s"></div>
            <div class="absolute inset-4 rounded-full bg-red-50/20 animate-ping" style="animation-delay:0.6s"></div>
          </div>
          <button @click="handleRecordClick" class="relative w-28 h-28 rounded-full flex items-center justify-center transition-all duration-300 active:scale-95"
            :class="translate.isRecording ? 'bg-gradient-to-br from-red-200 to-red-100 shadow-lg shadow-red-200/40' : 'bg-gradient-to-br from-[#B5D8CC] to-[#8DC4B0] shadow-lg shadow-[#B5D8CC]/30 hover:shadow-xl hover:-translate-y-0.5'">
            <Mic v-if="!translate.isRecording" :size="36" class="text-white" :stroke-width="1.5" />
            <Square v-else :size="28" class="text-red-400 fill-red-400" :stroke-width="2" />
          </button>
        </div>
        <!-- 待机状态：引导用户操作 -->
        <p v-if="!translate.isRecording" class="text-sm font-light tracking-wider text-[#8B9D8B]">
          点击录音，对着猫咪
        </p>
        <p v-if="!translate.isRecording" class="text-xs text-[#B5D8CC] mt-1 font-light">
          录 3-5 秒就好
        </p>
        <!-- 录音中：计时 -->
        <p v-if="translate.isRecording" class="text-sm font-light tracking-wider text-[#B5D8CC]">
          录音中 {{ formatTime(translate.recordingTime) }}
        </p>
        <p v-if="translate.isRecording" class="text-xs text-[#B5D8CC] mt-1 font-light">
          再次点击停止
        </p>
      </div>

      <!-- 错误提示 -->
      <div v-if="translate.errorMessage" class="mt-4 w-full max-w-sm px-4 py-3 rounded-xl bg-red-50/80 border border-red-100 flex items-center justify-between gap-3">
        <p class="text-xs text-red-400 font-light">{{ translate.errorMessage }}</p>
        <button @click="translate.errorMessage = ''" class="shrink-0 text-[10px] text-red-300 underline underline-offset-2">知道了</button>
      </div>

      <!-- 文件选择 -->
      <div class="text-center">
        <button @click="fileInput?.click()" class="inline-flex items-center gap-1.5 text-xs text-[#B5D8CC] hover:text-[#8B9D8B] transition-colors underline underline-offset-2 decoration-[#D4C4B0] font-light">
          <Upload :size="12" :stroke-width="1.5" />
          或选择音频文件
        </button>
        <input ref="fileInput" type="file" accept="audio/*" class="hidden" @change="handleFileSelect" />
      </div>

      <div class="mt-12 text-center">
        <p class="text-[10px] text-[#B5D8CC] font-light tracking-wider flex items-center justify-center gap-1">
          <Music :size="10" :stroke-width="1.5" />
          在安静环境下录制，识别更准确
        </p>
      </div>
    </div>

    <ResultCard v-if="translate.showResult && translate.currentResult" :result="translate.currentResult" @close="translate.dismissResult()" />
  </div>
</template>
