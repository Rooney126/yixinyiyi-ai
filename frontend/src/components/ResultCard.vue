<script setup lang="ts">
import { ref, computed } from 'vue'
import type { TranslationResult } from '../types'
import { X, ThumbsUp, ThumbsDown, Minus } from 'lucide-vue-next'

const props = defineProps<{ result: TranslationResult }>()
const emit = defineEmits<{ close: [] }>()

const feedbackSent = ref(false)

const pct = computed(() => props.result.confidence ? Math.round(props.result.confidence * 100) : 0)
const level = computed(() => props.result.confidence_level || 'high')

const EMOTION_COLORS: Record<string, { from: string, to: string }> = {
  Angry:       { from: 'from-[#F5E0D8]', to: 'to-[#EDD0C8]' },   // 暖红
  Defence:     { from: 'from-[#F5E0D8]', to: 'to-[#EDD0C8]' },
  Fighting:    { from: 'from-[#F5D8D8]', to: 'to-[#EDC8C8]' },   // 深暖红
  Warning:     { from: 'from-[#F5E0D8]', to: 'to-[#EDD0C8]' },
  Happy:       { from: 'from-[#EEF4F0]', to: 'to-[#E8F0EA]' },   // 绿（原来）
  Resting:     { from: 'from-[#EEF4F0]', to: 'to-[#E8F0EA]' },
  HuntingMind: { from: 'from-[#EEF4F0]', to: 'to-[#E8F0EA]' },
  MotherCall:  { from: 'from-[#EDE4F0]', to: 'to-[#E8DCEF]' },   // 紫
  Mating:      { from: 'from-[#EDE4F0]', to: 'to-[#E8DCEF]' },
  Paining:     { from: 'from-[#F0ECE8]', to: 'to-[#EBE5E0]' },   // 灰米
}
const moodBg = computed(() => EMOTION_COLORS[props.result.label || ''] || EMOTION_COLORS['Happy'])

const barColor = computed(() => {
  if (level.value === 'high') return 'bg-[#7CA895]'
  if (level.value === 'medium') return 'bg-amber-400'
  return 'bg-red-300'
})
const barColorText = computed(() => {
  if (level.value === 'high') return 'text-[#7CA895]'
  if (level.value === 'medium') return 'text-amber-500'
  return 'text-red-400'
})
const levelLabel = computed(() => {
  if (level.value === 'high') return '较高'
  if (level.value === 'medium') return '一般'
  return '较低'
})

async function sendFeedback(fb: 'good' | 'bad' | 'unsure') {
  if (feedbackSent.value) return
  feedbackSent.value = true
  try {
    const token = localStorage.getItem('auth_token') || ''
    const headers: Record<string, string> = { 'Content-Type': 'application/json' }
    if (token) headers['Authorization'] = `Bearer ${token}`
    await fetch('/api/translate/feedback', {
      method: 'POST',
      headers,
      body: JSON.stringify({
        result_id: props.result.id,
        label: props.result.label,
        confidence: props.result.confidence ?? 0,
        feedback: fb,
        timestamp: Date.now(),
      }),
    })
  } catch {}
}
</script>

<template>
  <transition name="card-up">
    <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/10 backdrop-blur-sm" @click.self="emit('close')">
      <div class="w-full max-w-md bg-[#FAFAF8] rounded-t-3xl sm:rounded-3xl shadow-2xl overflow-y-auto max-h-[85vh] animate-slide-up" style="-webkit-overflow-scrolling: touch; overscroll-behavior: contain">

        <!-- 顶部条 -->
        <div class="sticky top-0 z-10 flex items-center justify-between px-6 pt-5 pb-2 bg-[#FAFAF8]">
          <span class="text-[10px] text-[#B5D8CC] tracking-wider font-light">翻译结果</span>
          <button @click="emit('close')" class="w-7 h-7 rounded-full bg-[#E8E0D6] flex items-center justify-center text-[#8B9D8B] hover:bg-[#D4C4B0] transition">
            <X :size="14" :stroke-width="1.5" />
          </button>
        </div>

        <div class="px-6 pb-10 space-y-5">

          <!-- ── 情绪主卡片 ── -->
          <div class="rounded-2xl px-5 py-6 bg-gradient-to-b text-center" :class="[moodBg.from, moodBg.to]">
            <span class="text-3xl">{{ result.emotionEmoji || '🐱' }}</span>
            <p class="mt-2 text-xl text-[#3D3D3D] leading-relaxed font-light italic">「{{ result.translation }}」</p>
            <div class="mt-3">
              <span class="inline-block px-3 py-0.5 rounded-full text-[10px] text-[#7CA895] bg-white/60">{{ result.emotionState }}</span>
            </div>
          </div>

          <!-- ── 置信度（简化） ── -->
          <div class="flex items-center gap-3">
            <div class="flex-1 h-1 rounded-full bg-[#E8E0D6] overflow-hidden">
              <div class="h-full rounded-full transition-all duration-500" :class="barColor" :style="{ width: pct + '%' }"></div>
            </div>
            <span class="text-[10px] shrink-0" :class="barColorText + '/70'">{{ pct }}%</span>
          </div>

          <!-- ── 给你的建议 ── -->
          <div class="rounded-xl px-4 py-3 bg-[#F5E6D3]/50">
            <p class="text-xs text-[#7CA895] font-light mb-1">给你的建议</p>
            <p class="text-sm text-[#5C4E3E] leading-relaxed">{{ result.behaviorAdvice }}</p>
          </div>

          <!-- ── 详细（猫咪在说 + alternative） ── -->
          <div class="rounded-xl px-4 py-3 bg-white/70 border border-[#E8E0D6] space-y-2">
            <p class="text-xs text-[#7CA895] font-light">猫咪在说</p>
            <p class="text-sm text-[#5C4E3E] leading-relaxed">{{ result.catIntent }}</p>

            <div v-if="result.alternative" class="pt-2 border-t border-[#E8E0D6]/50">
              <p class="text-[10px] text-[#B5D8CC] mb-1 font-light">也可能是</p>
              <p class="text-sm text-[#5C4E3E] leading-relaxed">{{ result.alternative.translation }} <span class="text-[10px] text-[#B5D8CC]">（{{ Math.round((result.alternative.confidence ?? 0) * 100) }}%）</span></p>
            </div>
          </div>

          <!-- ── 反馈 ── -->
          <div class="flex items-center justify-center gap-3">
            <span class="text-[10px] text-[#B5D8CC] tracking-wider font-light">结果准吗？</span>
            <div class="flex gap-2">
              <button v-if="!feedbackSent" @click="sendFeedback('good')" class="w-8 h-8 rounded-full bg-[#EEF4F0] flex items-center justify-center hover:bg-[#D4E8DC] transition-colors">
                <ThumbsUp :size="14" class="text-[#7CA895]" :stroke-width="1.5" />
              </button>
              <button v-if="!feedbackSent" @click="sendFeedback('unsure')" class="w-8 h-8 rounded-full bg-[#F5E6D3] flex items-center justify-center hover:bg-[#E8D4BD] transition-colors">
                <Minus :size="14" class="text-[#B5956E]" :stroke-width="1.5" />
              </button>
              <button v-if="!feedbackSent" @click="sendFeedback('bad')" class="w-8 h-8 rounded-full bg-[#F5EEEE] flex items-center justify-center hover:bg-[#E8D4D4] transition-colors">
                <ThumbsDown :size="14" class="text-[#C46A6A]" :stroke-width="1.5" />
              </button>
              <span v-else class="text-[10px] text-[#7CA895] font-light self-center">已反馈 ✓</span>
            </div>
          </div>

        </div>
      </div>
    </div>
  </transition>
</template>

<style scoped>
.animate-slide-up { animation: slideUp 0.35s ease-out; }
@keyframes slideUp { from { transform: translateY(40px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
.card-up-leave-active { transition: all 0.2s ease; }
.card-up-leave-to { transform: translateY(20px); opacity: 0; }
</style>
