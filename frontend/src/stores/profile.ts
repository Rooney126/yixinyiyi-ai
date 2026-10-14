import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useProfileStore = defineStore('profile', () => {
  const catName = ref('')
  const avatar = ref('')
  const badges = ref<string[]>(['🐾 新手铲屎官'])

  const hasCat = computed(() => catName.value.trim().length > 0)

  function setCatName(name: string) {
    catName.value = name
    saveToLocal()
  }

  function loadFromLocal() {
    try {
      const data = localStorage.getItem('profile')
      if (data) {
        const p = JSON.parse(data)
        catName.value = p.catName || ''
        badges.value = p.badges || ['🐾 新手铲屎官']
      }
    } catch {}
  }

  function saveToLocal() {
    try {
      localStorage.setItem('profile', JSON.stringify({
        catName: catName.value,
        badges: badges.value,
      }))
    } catch {}
  }

  return {
    catName,
    avatar,
    badges,
    hasCat,
    setCatName,
    loadFromLocal,
  }
})
