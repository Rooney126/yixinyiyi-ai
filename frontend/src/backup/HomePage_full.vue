<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import type { HomeChannel } from '../types'
import { useTranslateStore } from '../stores/translate'
import {
  Mic, Heart, ShoppingBag, Users, Gamepad2,
  Cat, Dog, Stethoscope, Music, BookOpen, Pill, Lightbulb,
  ChevronRight, Star, Flame, Sparkles, Trophy,
  Fish, Bone, Cookie, Droplets, Scissors, Bed, Shirt, Apple,
  Radio, Camera, Crosshair, ToyBrick,
  MessageCircle, ArrowRight,
  Clock, Volume2, WandSparkles,
  PawPrint, Smile, Smartphone, Truck,
} from 'lucide-vue-next'

const router = useRouter()
const translate = useTranslateStore()
const activeChannel = ref<HomeChannel>('medical')
const communityTab = ref<'feed' | 'o2o'>('feed')

const channels = [
  { key: 'translate' as HomeChannel, label: '翻译', icon: Mic },
  { key: 'medical' as HomeChannel,   label: '医疗', icon: Heart },
  { key: 'market' as HomeChannel,    label: '商城', icon: ShoppingBag },
  { key: 'community' as HomeChannel, label: '社区', icon: Users },
  { key: 'games' as HomeChannel,     label: '游戏', icon: Gamepad2 },
]

const medicalFns = [
  { icon: Dog,    label: '宠物管家', desc: '档案管理' },
  { icon: Cat,    label: '爱宠小屋', desc: '环境监测' },
  { icon: WandSparkles, label: '宠物MBTI', desc: '性格测试' },
  { icon: Music,  label: '逗猫音乐', desc: '放松心情' },
  { icon: BookOpen, label: '喵星讲堂', desc: '养宠知识' },
  { icon: Pill,   label: '在线问诊', desc: '随时咨询' },
]

const marketItems = [
  { pic: Fish,      name: '天然无谷猫粮', price: 128, rating: 4.8, sales: '2.3万', tag: '热卖', tagIcon: Flame },
  { pic: ToyBrick,  name: '益智逗猫玩具', price: 69,  rating: 4.6, sales: '1.1万', tag: '新品', tagIcon: Sparkles },
  { pic: Bed,       name: '猫窝四季垫',   price: 199, rating: 4.9, sales: '3.2万', tag: '爆款', tagIcon: Trophy },
  { pic: Fish,      name: '冻干三文鱼',   price: 45,  rating: 4.7, sales: '5.6万', tag: '推荐', tagIcon: Star },
  { pic: Droplets,  name: '自动饮水机',   price: 159, rating: 4.5, sales: '8千',  tag: '',     tagIcon: null },
  { pic: Scissors,  name: '猫咪牙刷套装', price: 38,  rating: 4.3, sales: '6千',  tag: '',     tagIcon: null },
  { pic: Bone,      name: '鸡肉味猫条',   price: 29,  rating: 4.9, sales: '9.8万', tag: '热卖', tagIcon: Flame },
  { pic: Shirt,     name: '猫抓板剑麻',   price: 55,  rating: 4.7, sales: '4.5万', tag: '推荐', tagIcon: Star },
]

const marketCategories = [
  { label: '推荐', icon: Sparkles },
  { label: '猫粮', icon: Fish },
  { label: '玩具', icon: ToyBrick },
  { label: '日用', icon: Apple },
  { label: '医疗', icon: Pill },
  { label: '零食', icon: Cookie },
]
const marketCategory = ref(0)

const posts = [
  { name: '团团妈', time: '2小时前', text: '我家团团今天学会用猫砂了！老母亲欣慰', likes: 12, comments: 3, img: '/images/cat-1.jpg' },
  { name: '咪咪爸', time: '5小时前', text: '猫咪打喷嚏怎么办？需要去看医生吗？在线等急', likes: 8, comments: 5, img: '' },
  { name: '布丁姐姐', time: '昨天', text: '新买的猫窝它超喜欢，终于不睡快递盒了', likes: 24, comments: 12, img: '/images/cat-2.jpg' },
  { name: '豆豆妈', time: '昨天', text: '分享我家猫咪的每日食谱，长胖了 0.5kg！', likes: 31, comments: 7, img: '/images/cat-3.jpg' },
  { name: '小鱼干', time: '前天', text: '到家第一天，它躲在床底下不出来怎么办', likes: 45, comments: 23, img: '' },
  { name: '奶糖爸', time: '前天', text: '猫咪绝育后的护理心得，亲测有效', likes: 67, comments: 15, img: '' },
]

const games = [
  { icon: Crosshair, name: '逗猫棒',   desc: '屏幕互动逗猫', players: '3.2万' },
  { icon: ToyBrick,  name: '益智游戏', desc: '开发猫咪智力', players: '1.8万' },
  { icon: Music,     name: '猫语哼唱', desc: 'AI 生成猫曲',  players: '5.6万' },
  { icon: Camera,    name: '猫咪拍照', desc: '记录可爱瞬间', players: '4.2万' },
  { icon: Crosshair, name: '激光追猎', desc: '猫咪最爱光影', players: '2.7万' },
  { icon: Radio,     name: '抓线球',   desc: '简单小游戏',   players: '1.2万' },
]

const o2oServices = [
  { icon: Dog, label: '代遛', count: '326' },
  { icon: Sparkles, label: '代陪', count: '218' },
  { icon: Truck, label: '托运', count: '156' },
  { icon: Bed, label: '寄养', count: '534' },
  { icon: Cat, label: '领养', count: '89' },
  { icon: Scissors, label: '美容', count: '412' },
  { icon: Stethoscope, label: '上门医疗', count: '67' },
  { icon: Shirt, label: '宠物摄影', count: '143' },
]

const o2oProviders = [
  { name: '喵星人宠物生活馆', rating: 4.9, service: '寄养·美容·代遛', price: '¥68起' },
  { name: '爱宠无忧托运', rating: 4.8, service: '国内宠物托运', price: '¥200起' },
  { name: '暖心宠物寄养之家', rating: 4.7, service: '24h寄养·代陪', price: '¥88/天' },
  { name: '阳光宠物领养中心', rating: 4.9, service: '流浪猫领养', price: '免费' },
]

function goTranslate() {
  router.push('/translate')
}
</script>

<template>
  <div class="relative pb-4">
    <div class="absolute top-0 left-0 right-0 h-48 bg-gradient-to-b from-[#F5E6D3]/50 to-transparent pointer-events-none"></div>

    <!-- 频道标签 -->
    <div class="relative z-10 pt-3 pb-2 px-3 sticky top-0 bg-[#FAFAF8]/90 backdrop-blur-sm">
      <div class="flex gap-1.5 overflow-x-auto scrollbar-hide">
        <button v-for="ch in channels" :key="ch.key" @click="activeChannel = ch.key"
          class="shrink-0 flex items-center gap-1.5 px-5 py-2 rounded-full text-sm font-medium transition-all duration-300"
          :class="activeChannel === ch.key
            ? 'bg-white/80 backdrop-blur-sm text-[#5A7D6A] shadow-sm border border-[#D4C4B0]'
            : 'text-[#8B9D8B] hover:text-[#5A7D6A]'">
          <component :is="ch.icon" :size="14" :stroke-width="1.5" />
          {{ ch.label }}
        </button>
      </div>
    </div>

    <!-- ===== 医疗频道（默认） ===== -->
    <div v-if="activeChannel === 'medical'">
      <!-- 品牌区 -->
      <div class="px-6 pt-4 pb-5">
        <div class="relative bg-gradient-to-br from-[#F5E6D3] to-[#F0E8D8] rounded-3xl p-6 overflow-hidden shadow-sm">
          <div class="absolute inset-0 opacity-[0.04]">
            <PawPrint :size="48" :stroke-width="1" class="absolute top-3 left-4 text-[#B5D8CC]" />
            <PawPrint :size="36" :stroke-width="1" class="absolute bottom-2 right-6 text-[#B5D8CC]" />
          </div>
          <div class="relative flex items-center gap-4">
            <div class="shrink-0 w-16 h-16 rounded-2xl bg-white/70 backdrop-blur flex items-center justify-center shadow-inner">
              <Cat :size="32" :stroke-width="1.5" class="text-[#B5D8CC]" />
            </div>
            <div class="flex-1">
              <h2 class="text-lg font-light text-[#5C4E3E] tracking-wide">今天你的猫说了什么？</h2>
              <p class="text-xs text-[#8B9D8B] mt-1 font-light">轻轻一点，听懂它的心事</p>
            </div>
          </div>
        </div>
        <!-- 健康提醒 -->
        <div class="mt-3 bg-[#B5D8CC]/15 rounded-2xl px-4 py-3 flex items-center gap-3 border border-[#B5D8CC]/30">
          <Lightbulb :size="18" :stroke-width="1.5" class="text-[#B5D8CC]" />
          <p class="text-xs text-[#5A7D6A] font-light leading-relaxed">春季换毛期，建议每天梳理毛发，预防毛球症</p>
        </div>
      </div>

      <!-- 快捷入口 -->
      <div class="px-6 pb-4">
        <div class="flex items-center justify-between mb-3">
          <h3 class="text-sm text-[#8B9D8B] tracking-wider font-light">常用功能</h3>
          <span class="text-[10px] text-[#D4C4B0] font-light flex items-center gap-1">
            查看全部 <ArrowRight :size="10" :stroke-width="1.5" />
          </span>
        </div>
        <div class="grid grid-cols-3 gap-3">
          <div v-for="item in medicalFns" :key="item.label"
            class="bg-white/70 rounded-2xl p-3.5 text-center shadow-sm border border-[#E8E0D6] hover:bg-white/90 transition cursor-pointer active:scale-95">
            <component :is="item.icon" :size="20" :stroke-width="1.5" class="mx-auto mb-1.5 text-[#B5D8CC]" />
            <span class="text-xs text-[#5C4E3E] block">{{ item.label }}</span>
            <span class="text-[10px] text-[#D4C4B0] font-light">{{ item.desc }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- ===== 翻译频道 ===== -->
    <div v-if="activeChannel === 'translate'" class="px-6 pt-4 space-y-4">
      <button @click="goTranslate"
        class="relative w-full overflow-hidden rounded-2xl bg-gradient-to-r from-[#B5D8CC] to-[#8DC4B0] px-6 py-5 text-white shadow-lg shadow-[#B5D8CC]/30 transition-all duration-300 hover:shadow-xl active:scale-[0.98]">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <Mic :size="24" :stroke-width="1.5" />
            <div class="text-left">
              <p class="text-sm font-medium tracking-wider">开始翻译</p>
              <p class="text-[10px] text-white/70 mt-0.5 font-light">点击录音，最长 15 秒</p>
            </div>
          </div>
          <ChevronRight :size="20" :stroke-width="1.5" class="opacity-60" />
        </div>
      </button>

      <!-- 翻译历史预览 -->
      <div>
        <div class="flex items-center justify-between mb-3">
          <h3 class="text-sm text-[#8B9D8B] tracking-wider font-light">最近翻译</h3>
          <span class="text-[10px] text-[#D4C4B0] font-light">查看更多 →</span>
        </div>
        <div class="space-y-2">
          <div v-for="m in translate.memories.slice(0, 3)" :key="m.id"
            class="bg-white/70 rounded-2xl px-4 py-3 shadow-sm border border-[#E8E0D6] flex items-center gap-3">
            <Smile :size="22" :stroke-width="1.5" class="text-[#B5D8CC]" />
            <div class="flex-1 min-w-0">
              <p class="text-sm text-[#5C4E3E] truncate">{{ m.result.catIntent }}</p>
              <p class="text-[10px] text-[#D4C4B0] font-light">{{ new Date(m.timestamp).toLocaleString('zh-CN') }}</p>
            </div>
            <span class="text-xs font-medium" :class="m.result.emotionScore >= 7 ? 'text-red-400' : 'text-[#8B9D8B]'">{{ m.result.emotionScore }}/10</span>
          </div>
          <p v-if="translate.memories.length === 0" class="text-xs text-[#D4C4B0] text-center py-4 font-light">还没有翻译记录</p>
        </div>
      </div>

      <!-- 小贴士 -->
      <div class="bg-white/50 rounded-2xl p-4 border border-[#E8E0D6]">
        <h4 class="text-xs text-[#8B9D8B] tracking-wider mb-3 font-light">翻译小贴士</h4>
        <div class="grid grid-cols-3 gap-2 text-center text-[10px] text-[#5A7D6A]">
          <div class="bg-[#B5D8CC]/10 rounded-xl py-2.5">
            <Volume2 :size="18" :stroke-width="1.5" class="mx-auto mb-0.5 text-[#B5D8CC]" />
            安静环境
          </div>
          <div class="bg-[#B5D8CC]/10 rounded-xl py-2.5">
            <Clock :size="18" :stroke-width="1.5" class="mx-auto mb-0.5 text-[#B5D8CC]" />
            15秒足够
          </div>
          <div class="bg-[#B5D8CC]/10 rounded-xl py-2.5">
            <Smartphone :size="18" :stroke-width="1.5" class="mx-auto mb-0.5 text-[#B5D8CC]" />
            多录几次
          </div>
        </div>
      </div>
    </div>

    <!-- ===== 商城频道 ===== -->
    <div v-if="activeChannel === 'market'" class="px-6 pt-4 space-y-4">
      <!-- 分类标签 -->
      <div class="flex gap-2 overflow-x-auto scrollbar-hide pb-1">
        <button v-for="(cat, i) in marketCategories" :key="cat.label"
          @click="marketCategory = i"
          class="shrink-0 flex items-center gap-1 px-3.5 py-1.5 rounded-full text-xs font-medium transition-all duration-200 whitespace-nowrap"
          :class="marketCategory === i
            ? 'bg-[#B5D8CC] text-white'
            : 'bg-white/70 text-[#8B9D8B] border border-[#E8E0D6]'">
          <component :is="cat.icon" :size="12" :stroke-width="1.5" />
          {{ cat.label }}
        </button>
      </div>

      <!-- 热门推荐 -->
      <div class="bg-gradient-to-br from-[#F5E6D3] to-[#F0E8D8] rounded-2xl p-4 flex items-center gap-4 shadow-sm">
        <Trophy :size="32" :stroke-width="1" class="text-[#B5D8CC]" />
        <div class="flex-1">
          <p class="text-sm text-[#5C4E3E] font-medium">萌宠好物榜</p>
          <p class="text-[10px] text-[#8B9D8B] mt-0.5 font-light">本周热卖 100 件</p>
        </div>
        <ChevronRight :size="16" :stroke-width="1.5" class="text-[#D4C4B0]" />
      </div>

      <!-- 商品列表 -->
      <div class="grid grid-cols-2 gap-3">
        <div v-for="item in marketItems" :key="item.name"
          class="bg-white/80 rounded-2xl overflow-hidden shadow-sm border border-[#E8E0D6] hover:shadow-md transition cursor-pointer">
          <div class="h-24 bg-gradient-to-br from-[#F5E6D3]/40 to-[#FAFAF8] flex items-center justify-center relative">
            <component :is="item.pic" :size="30" :stroke-width="1.5" class="text-[#B5D8CC]" />
            <span v-if="item.tag" class="absolute top-1.5 left-1.5 px-1.5 py-0.5 rounded-md text-[8px] font-medium bg-[#B5D8CC]/20 text-[#5A7D6A] flex items-center gap-0.5">
              <component v-if="item.tagIcon" :is="item.tagIcon" :size="8" :stroke-width="1.5" />
              {{ item.tag }}
            </span>
          </div>
          <div class="p-2.5">
            <p class="text-xs text-[#5C4E3E] truncate">{{ item.name }}</p>
            <div class="flex items-center justify-between mt-1.5">
              <span class="text-sm font-medium text-[#5A7D6A]">¥{{ item.price }}</span>
              <span class="text-[10px] text-[#D4C4B0] font-light">已售 {{ item.sales }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ===== 社区频道 ===== -->
    <div v-if="activeChannel === 'community'" class="px-6 pt-4 space-y-5">
      <!-- 顶部切换标签 -->
      <div class="flex gap-3 border-b border-[#E8E0D6] pb-2">
        <button @click="communityTab = 'feed'" class="text-xs font-medium pb-2 -mb-0.5 transition-colors border-b-2" :class="communityTab === 'feed' ? 'text-[#5A7D6A] border-[#B5D8CC]' : 'text-[#D4C4B0] border-transparent'">
          <Users :size="14" :stroke-width="1.5" class="inline mr-1" />宠友圈
        </button>
        <button @click="communityTab = 'o2o'" class="text-xs font-medium pb-2 -mb-0.5 transition-colors border-b-2" :class="communityTab === 'o2o' ? 'text-[#5A7D6A] border-[#B5D8CC]' : 'text-[#D4C4B0] border-transparent'">
          <Sparkles :size="14" :stroke-width="1.5" class="inline mr-1" />宠物服务
        </button>
      </div>

      <!-- ===== 宠友圈 ===== -->
      <template v-if="communityTab === 'feed'">
        <!-- 话题标签 -->
        <div class="flex gap-2 overflow-x-auto scrollbar-hide pb-1">
          <span v-for="tag in ['#经验分享', '#晒猫', '#养宠求助', '#猫粮测评', '#医疗科普']" :key="tag"
            class="shrink-0 px-3 py-1 rounded-full text-[10px] text-[#5A7D6A] bg-[#B5D8CC]/15 border border-[#B5D8CC]/20 font-light whitespace-nowrap">
            {{ tag }}
          </span>
        </div>

        <!-- 发帖入口 -->
        <div class="bg-white/70 rounded-2xl p-3 shadow-sm border border-[#E8E0D6] flex items-center gap-3 cursor-pointer hover:bg-white/90 transition">
          <div class="w-8 h-8 rounded-full bg-[#B5D8CC]/30 flex items-center justify-center">
            <Cat :size="16" :stroke-width="1.5" class="text-[#B5D8CC]" />
          </div>
          <p class="text-xs text-[#D4C4B0] font-light flex-1">分享你和猫咪的故事…</p>
          <button class="px-4 py-1.5 rounded-full bg-[#B5D8CC] text-white text-[10px] font-medium hover:bg-[#8DC4B0] transition">发布</button>
        </div>

        <!-- 晒猫专区标题 -->
        <div class="flex items-center justify-between">
          <h3 class="text-xs text-[#8B9D8B] font-light tracking-wider flex items-center gap-1">📸 今日晒猫</h3>
          <span class="text-[10px] text-[#D4C4B0] font-light">更多 →</span>
        </div>
        <div class="grid grid-cols-3 gap-2">
          <div v-for="(img, i) in ['/images/cat-1.jpg', '/images/cat-2.jpg', '/images/cat-3.jpg']" :key="i" class="aspect-square rounded-xl overflow-hidden bg-[#F5E6D3]/30">
            <img :src="img" class="w-full h-full object-cover" loading="lazy" />
          </div>
        </div>

        <!-- 经验分享帖子 -->
        <div class="flex items-center justify-between mt-1">
          <h3 class="text-xs text-[#8B9D8B] font-light tracking-wider flex items-center gap-1">💬 经验分享</h3>
        </div>
        <div v-for="(post, i) in posts" :key="i" class="bg-white/80 rounded-2xl overflow-hidden shadow-sm border border-[#E8E0D6]">
          <div class="p-4 pb-3">
            <div class="flex items-center gap-2.5 mb-2.5">
              <div class="w-7 h-7 rounded-full bg-[#F5E6D3] flex items-center justify-center">
                <Cat :size="14" :stroke-width="1.5" class="text-[#B5D8CC]" />
              </div>
              <div class="flex-1">
                <p class="text-xs text-[#5C4E3E] font-medium">{{ post.name }}</p>
                <p class="text-[10px] text-[#D4C4B0] font-light">{{ post.time }}</p>
              </div>
              <span class="text-[10px] text-[#5A7D6A] bg-[#B5D8CC]/10 px-2 py-0.5 rounded-full">{{ ['经验','求助','分享','日记','心得','问答'][i] }}</span>
            </div>
            <p class="text-sm text-[#6B5D4D] leading-relaxed font-light">{{ post.text }}</p>
            <div v-if="post.img && i === 0" class="mt-2.5 -mx-2">
              <img :src="post.img" class="w-full h-32 object-cover rounded-xl" loading="lazy" />
            </div>
          </div>
          <div class="px-4 pb-3 flex items-center gap-4 text-xs text-[#D4C4B0] border-t border-[#FAFAF8] pt-2.5">
            <span class="flex items-center gap-1 cursor-pointer hover:text-[#5A7D6A] transition"><Heart :size="12" :stroke-width="1.5" /> {{ post.likes }}</span>
            <span class="flex items-center gap-1 cursor-pointer hover:text-[#5A7D6A] transition"><MessageCircle :size="12" :stroke-width="1.5" /> {{ post.comments }}</span>
            <span class="flex items-center gap-1 cursor-pointer hover:text-[#5A7D6A] transition ml-auto"><Star :size="10" :stroke-width="1.5" /> 收藏</span>
          </div>
        </div>
      </template>

      <!-- ===== O2O 服务平台 ===== -->
      <template v-if="communityTab === 'o2o'">
        <div class="bg-gradient-to-br from-[#B5D8CC]/10 to-[#F5E6D3]/20 rounded-2xl p-4 border border-[#E8E0D6]">
          <h3 class="text-sm text-[#5C4E3E] font-medium flex items-center gap-2"><Sparkles :size="16" :stroke-width="1.5" class="text-[#B5D8CC]" /> 专业宠物服务</h3>
          <p class="text-[10px] text-[#8B9D8B] font-light mt-1">严选服务商 · 在线预约 · 安心保障</p>
        </div>

        <!-- 服务分类网格 -->
        <div class="grid grid-cols-4 gap-3">
          <div v-for="svc in o2oServices" :key="svc.label" class="bg-white/70 rounded-2xl p-3 text-center shadow-sm border border-[#E8E0D6] hover:bg-white/90 transition cursor-pointer active:scale-95">
            <component :is="svc.icon" :size="22" :stroke-width="1.5" class="mx-auto mb-1.5 text-[#B5D8CC]" />
            <span class="text-[10px] text-[#5C4E3E] block">{{ svc.label }}</span>
            <span class="text-[8px] text-[#D4C4B0] font-light">{{ svc.count }}人</span>
          </div>
        </div>

        <!-- 推荐服务商 -->
        <h3 class="text-xs text-[#8B9D8B] font-light tracking-wider">推荐服务商</h3>
        <div v-for="s in o2oProviders" :key="s.name" class="bg-white/80 rounded-2xl p-4 shadow-sm border border-[#E8E0D6] flex items-center gap-3">
          <div class="w-12 h-12 rounded-xl bg-[#B5D8CC]/20 flex items-center justify-center shrink-0">
            <Cat :size="22" :stroke-width="1.5" class="text-[#B5D8CC]" />
          </div>
          <div class="flex-1 min-w-0">
            <div class="flex items-center gap-2">
              <p class="text-sm text-[#5C4E3E] font-medium">{{ s.name }}</p>
              <span class="text-[10px] text-[#8B9D8B]">{{ s.rating }} ⭐</span>
            </div>
            <p class="text-[10px] text-[#D4C4B0] font-light mt-0.5">{{ s.service }} · {{ s.price }}</p>
          </div>
          <button class="px-3 py-1.5 rounded-full bg-[#B5D8CC] text-white text-[10px] font-medium hover:bg-[#8DC4B0] transition shrink-0">预约</button>
        </div>
      </template>
    </div>

    <!-- ===== 游戏频道 ===== -->
    <div v-if="activeChannel === 'games'" class="px-6 pt-4 space-y-4">
      <!-- 推荐游戏横幅 -->
      <div class="bg-gradient-to-br from-[#B5D8CC]/20 to-[#F5E6D3]/40 rounded-2xl p-5 shadow-sm border border-[#E8E0D6]">
        <div class="flex items-center gap-4">
          <Crosshair :size="32" :stroke-width="1.5" class="text-[#B5D8CC]" />
          <div class="flex-1">
            <p class="text-sm text-[#5C4E3E] font-medium">逗猫棒</p>
            <p class="text-[10px] text-[#8B9D8B] mt-0.5 font-light">3.2 万铲屎官在玩 · 互动率最高</p>
          </div>
          <button class="px-4 py-1.5 rounded-full bg-[#B5D8CC] text-white text-[10px] font-medium hover:bg-[#8DC4B0] transition">开始</button>
        </div>
      </div>

      <h3 class="text-sm text-[#8B9D8B] tracking-wider font-light">全部游戏</h3>
      <div class="grid grid-cols-2 gap-3">
        <div v-for="g in games" :key="g.name"
          class="bg-white/70 rounded-2xl p-4 shadow-sm border border-[#E8E0D6] hover:bg-white/90 transition cursor-pointer active:scale-95">
          <component :is="g.icon" :size="24" :stroke-width="1.5" class="mb-2.5 text-[#B5D8CC]" />
          <p class="text-sm text-[#5C4E3E]">{{ g.name }}</p>
          <p class="text-[10px] text-[#D4C4B0] mt-0.5 font-light">{{ g.desc }}</p>
          <p class="text-[10px] text-[#5A7D6A] mt-1.5 font-light">{{ g.players }} 人在玩</p>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.scrollbar-hide { -ms-overflow-style: none; scrollbar-width: none; }
.scrollbar-hide::-webkit-scrollbar { display: none; }
</style>
