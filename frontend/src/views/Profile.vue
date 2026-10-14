<script setup lang="ts">
import { ref, computed } from 'vue'
import { useProfileStore } from '../stores/profile'
import { useTranslateStore } from '../stores/translate'
import {
  MessageSquare, BookOpen, Receipt, Hospital, Gem, Heart, Image,
  Bell, FileText, FileSignature, LogOut, Phone, ChevronRight,
  Cat, Award, Pencil, X, BookMarked,
} from 'lucide-vue-next'

const profile = useProfileStore()
const translate = useTranslateStore()
const editingName = ref(false)
const nameInput = ref('')

function startEdit() { nameInput.value = profile.catName; editingName.value = true }
function saveName() {
  if (nameInput.value.trim()) { profile.setCatName(nameInput.value.trim()); translate.initDemoMemories(nameInput.value.trim()) }
  editingName.value = false
}

const featureItems = [
  { icon: MessageSquare, label: '消息', badge: '' },
  { icon: BookOpen, label: '回忆录', badge: `${translate.memories.length}` },
  { icon: Receipt, label: '交易记录', badge: '' },
  { icon: Hospital, label: '问诊记录', badge: '' },
  { icon: Gem, label: 'VIP爱宠区', badge: '' },
  { icon: Heart, label: '收藏', badge: '' },
  { icon: Image, label: '作品', badge: '' },
]

const systemItems = [
  { icon: Bell, label: '通知管理' },
  { icon: FileText, label: '隐私协议' },
  { icon: FileSignature, label: '会员续订协议' },
  { icon: FileSignature, label: '用户协议' },
  { icon: LogOut, label: '退出登录' },
  { icon: Phone, label: '联系我们' },
]

const showMemories = ref(false)
const memories = computed(() => translate.memories)
</script>

<template>
  <div class="pb-4">
    <div class="relative pt-12 pb-8 px-6">
      <div class="absolute top-0 left-0 right-0 h-48 bg-gradient-to-b from-[#F5E6D3]/50 to-transparent rounded-b-[2rem]"></div>
      <div class="relative flex flex-col items-center">
        <div class="relative mb-4">
          <div class="w-20 h-20 rounded-full bg-gradient-to-br from-[#F5E6D3] to-[#F0E8D8] flex items-center justify-center shadow-inner shadow-[#B5D8CC]/20">
            <Cat :size="36" :stroke-width="1.5" class="text-[#B5D8CC]" />
          </div>
          <div class="absolute -bottom-1 -right-1 w-6 h-6 rounded-full bg-white shadow-sm flex items-center justify-center">
            <Award :size="14" :stroke-width="1.5" class="text-[#B5D8CC]" />
          </div>
        </div>
        <div v-if="editingName" class="flex items-center gap-2 mb-2">
          <input v-model="nameInput" placeholder="给猫咪取个名字" maxlength="12" class="px-4 py-2 border border-[#D4C4B0] rounded-xl text-sm text-center outline-none focus:border-[#B5D8CC] bg-white/80 w-44 text-[#5C4E3E]" @keyup.enter="saveName" />
          <button @click="saveName" class="w-8 h-8 rounded-xl bg-[#B5D8CC] text-white text-xs flex items-center justify-center hover:bg-[#8DC4B0] transition">
            <X :size="14" :stroke-width="2" />
          </button>
        </div>
        <div v-else class="flex items-center gap-2 mb-1">
          <span class="text-lg font-medium text-[#5C4E3E]">{{ profile.catName || '未取名' }}</span>
          <button @click="startEdit" class="text-[#B5D8CC] hover:text-[#8B9D8B] transition">
            <Pencil :size="14" :stroke-width="1.5" />
          </button>
        </div>
        <div class="flex gap-2">
          <span v-for="badge in profile.badges" :key="badge" class="px-3 py-0.5 rounded-full bg-[#F5E6D3] text-xs text-[#8B9D8B] font-light flex items-center gap-1">
            <Award :size="10" :stroke-width="1.5" class="text-[#B5D8CC]" />
            {{ badge.replace('🐾 ', '') }}
          </span>
        </div>
      </div>
    </div>

    <div class="px-5 space-y-6">
      <div class="bg-white/70 backdrop-blur rounded-2xl shadow-sm border border-[#D4C4B0] overflow-hidden divide-y divide-[#F5E6D3]">
        <div v-for="item in featureItems" :key="item.label" class="flex items-center justify-between px-5 py-3.5 cursor-pointer hover:bg-[#F5E6D3]/30 transition active:bg-[#F5E6D3]/50" @click="item.label === '回忆录' ? (showMemories = true) : null">
          <div class="flex items-center gap-3">
            <component :is="item.icon" :size="18" :stroke-width="1.5" class="text-[#8B9D8B]" />
            <span class="text-sm text-[#5C4E3E]">{{ item.label }}</span>
          </div>
          <div class="flex items-center gap-2">
            <span v-if="item.badge" class="px-2 py-0.5 rounded-full bg-[#F5E6D3] text-xs text-[#8B9D8B]">{{ item.badge }}</span>
            <ChevronRight :size="14" :stroke-width="1.5" class="text-[#D4C4B0]" />
          </div>
        </div>
      </div>

      <div>
        <p class="text-xs text-[#B5D8CC] tracking-wider px-5 mb-2 font-light">系统</p>
        <div class="bg-white/50 backdrop-blur rounded-2xl shadow-sm border border-[#D4C4B0] overflow-hidden divide-y divide-[#F5E6D3]">
          <div v-for="item in systemItems" :key="item.label" class="flex items-center gap-3 px-5 py-3.5 cursor-pointer hover:bg-[#F5E6D3]/30 transition">
            <component :is="item.icon" :size="16" :stroke-width="1.5" class="text-[#B5D8CC]" />
            <span class="text-sm text-[#4A5D4A] font-light">{{ item.label }}</span>
          </div>
        </div>
      </div>

      <p class="text-center text-[10px] text-[#D4C4B0] tracking-widest py-4 font-light">译芯怡意 · 听懂你的猫</p>
    </div>

    <!-- 回忆录弹窗 -->
    <transition name="fade">
      <div v-if="showMemories" class="fixed inset-0 z-50 bg-black/10 backdrop-blur-sm flex items-end sm:items-center justify-center" @click.self="showMemories = false">
        <div class="w-full max-w-md bg-[#FAFAF8] rounded-t-3xl sm:rounded-3xl shadow-2xl max-h-[70vh] overflow-hidden flex flex-col">
          <div class="flex justify-between items-center px-6 pt-6 pb-3">
            <h3 class="text-base text-[#5C4E3E] font-medium flex items-center gap-2">
              <BookMarked :size="18" :stroke-width="1.5" class="text-[#B5D8CC]" />
              回忆录
            </h3>
            <button @click="showMemories = false" class="w-8 h-8 rounded-full bg-[#E8E0D6] flex items-center justify-center text-[#8B9D8B] hover:bg-[#D4C4B0] transition">
              <X :size="14" :stroke-width="1.5" />
            </button>
          </div>
          <div class="px-6 pb-6 overflow-y-auto space-y-3 flex-1">
            <div v-for="m in memories" :key="m.id" class="bg-white/80 backdrop-blur rounded-2xl p-4 shadow-sm border border-[#D4C4B0]">
              <div class="flex items-center justify-between mb-2">
                <span class="text-[10px] text-[#B5D8CC]">{{ new Date(m.timestamp).toLocaleString('zh-CN') }}</span>
                <Smile :size="18" :stroke-width="1.5" class="text-[#B5D8CC]" />
              </div>
              <p class="text-sm text-[#4A5D4A] leading-relaxed font-light">{{ m.result.catIntent }}</p>
              <div class="flex items-center gap-2 mt-2">
                <span class="text-[10px] text-[#B5D8CC] font-light">情绪评分</span>
                <span class="text-sm font-medium" :class="m.result.emotionScore >= 7 ? 'text-red-400' : m.result.emotionScore >= 5 ? 'text-amber-400' : 'text-[#7CA895]'">{{ m.result.emotionScore }}/10</span>
              </div>
            </div>
            <p v-if="memories.length === 0" class="text-center text-[#B5D8CC] text-sm py-8 font-light">还没有翻译记录</p>
          </div>
        </div>
      </div>
    </transition>
  </div>
</template>

<style scoped>
.fade-enter-active, .fade-leave-active { transition: opacity 0.2s ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
</style>
