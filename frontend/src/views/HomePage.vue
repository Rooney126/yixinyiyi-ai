<script setup lang="ts">
import { useRouter } from 'vue-router'
import { useTranslateStore } from '../stores/translate'
import { Mic, ChevronRight, Smile, Clock, Volume2, Smartphone, Cat } from 'lucide-vue-next'

const router = useRouter()
const translate = useTranslateStore()

function goTranslate() {
  router.push('/translate')
}
</script>

<template>
  <div class="flex flex-col items-center min-h-full pt-12 pb-4 relative">
    <div class="absolute top-0 left-0 right-0 h-48 bg-gradient-to-b from-[#F5E6D3]/40 to-transparent pointer-events-none"></div>

    <div class="relative z-10 w-full px-6 flex flex-col items-center">
      <!-- 品牌区 -->
      <div class="text-center mb-8">
        <div class="w-16 h-16 rounded-2xl bg-gradient-to-br from-[#B5D8CC] to-[#8DC4B0] flex items-center justify-center mx-auto mb-4 shadow-lg">
          <Cat :size="32" :stroke-width="1.5" class="text-white" />
        </div>
        <h1 class="text-lg font-light text-[#5C4E3E] tracking-wide">译芯怡意</h1>
        <p class="text-xs text-[#8B9D8B] mt-1 font-light">AI 猫语翻译</p>
      </div>

      <!-- 翻译入口 -->
      <button @click="goTranslate"
        class="w-full rounded-2xl bg-gradient-to-r from-[#B5D8CC] to-[#8DC4B0] px-6 py-5 text-white shadow-lg shadow-[#B5D8CC]/30 transition-all duration-300 hover:shadow-xl active:scale-[0.98]">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <Mic :size="24" :stroke-width="1.5" />
            <div class="text-left">
              <p class="text-sm font-medium tracking-wider">开始翻译</p>
              <p class="text-[10px] text-white/70 mt-0.5 font-light">点击录音，3-5秒就够了</p>
            </div>
          </div>
          <ChevronRight :size="20" :stroke-width="1.5" class="opacity-60" />
        </div>
      </button>

      <!-- 翻译历史 -->
      <div class="w-full mt-8">
        <h3 class="text-sm text-[#8B9D8B] tracking-wider font-light mb-3">最近翻译</h3>
        <div class="space-y-2">
          <div v-for="m in translate.memories.slice(0, 5)" :key="m.id"
            class="bg-white/70 rounded-2xl px-4 py-3 shadow-sm border border-[#E8E0D6] flex items-center gap-3">
            <Smile :size="20" :stroke-width="1.5" class="text-[#B5D8CC]" />
            <div class="flex-1 min-w-0">
              <p class="text-sm text-[#5C4E3E] truncate">{{ m.result.catIntent }}</p>
              <p class="text-[10px] text-[#D4C4B0] font-light">{{ new Date(m.timestamp).toLocaleString('zh-CN') }}</p>
            </div>
          </div>
          <p v-if="translate.memories.length === 0" class="text-xs text-[#D4C4B0] text-center py-4 font-light">
            还没有翻译记录
          </p>
        </div>
      </div>

      <!-- 小贴士 -->
      <div class="w-full mt-6 bg-white/50 rounded-2xl p-4 border border-[#E8E0D6]">
        <h4 class="text-xs text-[#8B9D8B] tracking-wider mb-3 font-light">翻译小贴士</h4>
        <div class="grid grid-cols-3 gap-2 text-center text-[10px] text-[#5A7D6A]">
          <div class="bg-[#B5D8CC]/10 rounded-xl py-2.5">
            <Volume2 :size="18" :stroke-width="1.5" class="mx-auto mb-0.5 text-[#B5D8CC]" />
            安静环境
          </div>
          <div class="bg-[#B5D8CC]/10 rounded-xl py-2.5">
            <Clock :size="18" :stroke-width="1.5" class="mx-auto mb-0.5 text-[#B5D8CC]" />
            3-5秒足够
          </div>
          <div class="bg-[#B5D8CC]/10 rounded-xl py-2.5">
            <Smartphone :size="18" :stroke-width="1.5" class="mx-auto mb-0.5 text-[#B5D8CC]" />
            多录几次
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
