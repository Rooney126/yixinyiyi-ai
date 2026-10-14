<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { Cat, PawPrint } from 'lucide-vue-next'

const emit = defineEmits<{ done: [] }>()
const show = ref(true)

onMounted(() => {
  setTimeout(() => {
    show.value = false
    emit('done')
  }, 500)
})
</script>

<template>
  <transition name="splash-fade">
    <div v-if="show" class="fixed inset-0 z-[9999] flex flex-col items-center justify-center bg-[#FAFAF8]">
      <div class="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-48 bg-[#F5E6D3]/40 rounded-full blur-3xl"></div>
      <div class="relative mb-8">
        <div class="w-24 h-24 rounded-[1.4rem] bg-white shadow-lg shadow-[#B5D8CC]/20 flex items-center justify-center">
          <Cat :size="48" :stroke-width="1.5" class="text-[#B5D8CC]" />
        </div>
        <div class="absolute -top-1 -right-1 w-3 h-3 rounded-full bg-[#B5D8CC]/30"></div>
        <div class="absolute -bottom-1 -left-1 w-2 h-2 rounded-full bg-[#D4C4B0]/40"></div>
      </div>
      <h1 class="text-xl font-light tracking-[0.3em] text-[#5C4E3E] mb-2">译芯怡意</h1>
      <p class="text-xs tracking-[0.2em] text-[#B5D8CC] font-light">听·懂·你·的·猫</p>
      <div class="absolute bottom-[15%] flex gap-4">
        <PawPrint :size="20" :stroke-width="1" class="opacity-30 animate-bounce text-[#B5D8CC]" style="animation-delay: 0s" />
        <PawPrint :size="20" :stroke-width="1" class="opacity-30 animate-bounce text-[#B5D8CC]" style="animation-delay: 0.2s" />
        <PawPrint :size="20" :stroke-width="1" class="opacity-30 animate-bounce text-[#B5D8CC]" style="animation-delay: 0.4s" />
      </div>
    </div>
  </transition>
</template>

<style scoped>
.splash-fade-leave-active { transition: opacity 0.5s ease; }
.splash-fade-leave-to { opacity: 0; }
</style>
