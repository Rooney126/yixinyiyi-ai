<script setup lang="ts">
import { ref } from 'vue'
import { RouterView, useRoute, useRouter } from 'vue-router'
import SplashScreen from './components/SplashScreen.vue'
import { House, Mic, User } from 'lucide-vue-next'

const route = useRoute()
const router = useRouter()
const showSplash = ref(true)

function onSplashDone() {
  showSplash.value = false
}

const tabs = [
  { path: '/',        icon: House,   label: '首页' },
  { path: '/translate', icon: Mic,   label: '翻译' },
  { path: '/profile', icon: User,   label: '我的' },
]

function isActive(path: string) {
  if (path === '/') return route.path === '/'
  return route.path.startsWith(path)
}

function switchTab(path: string) {
  router.push(path)
}
</script>

<template>
  <SplashScreen v-if="showSplash" @done="onSplashDone" />

  <div v-show="!showSplash" class="h-screen flex flex-col bg-[#FAFAF8] overflow-hidden">
    <main class="flex-1 overflow-y-auto pb-16 scroll-smooth">
      <RouterView />
    </main>

    <!-- 底部导航 -->
    <nav class="fixed bottom-0 left-0 right-0 z-50">
      <div class="bg-white/90 backdrop-blur-xl border-t border-[#D4C4B0] safe-bottom">
        <div class="flex justify-around items-center max-w-md mx-auto px-2">
          <button
            v-for="tab in tabs"
            :key="tab.path"
            @click="switchTab(tab.path)"
            class="relative flex flex-col items-center gap-0.5 py-2.5 px-5 transition-all duration-300"
          >
            <div v-if="isActive(tab.path)" class="absolute -top-0.5 left-1/2 -translate-x-1/2 w-8 h-1 bg-[#B5D8CC] rounded-full"></div>
            <component :is="tab.icon" :size="20" :stroke-width="1.5" class="transition-transform duration-200" :class="isActive(tab.path) ? 'text-[#7D9D8D] scale-110' : 'text-[#B5D8CC]'" />
            <span class="text-[10px] font-medium tracking-wider transition-colors duration-200" :class="isActive(tab.path) ? 'text-[#7D9D8D]' : 'text-[#B5D8CC]/50'">
              {{ tab.label }}
            </span>
          </button>
        </div>
      </div>
    </nav>
  </div>
</template>

<style>
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'PingFang SC', 'Noto Sans SC', sans-serif;
  -webkit-font-smoothing: antialiased;
  background-color: #FAFAF8;
  -webkit-tap-highlight-color: transparent;
}
.safe-bottom { padding-bottom: env(safe-area-inset-bottom, 0); }
.scroll-smooth { scroll-behavior: smooth; }
.fade-enter-active, .fade-leave-active { transition: opacity 0.2s ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
</style>
