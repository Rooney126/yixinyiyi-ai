import { createRouter, createWebHistory } from 'vue-router'
import HomePage from '../views/HomePage.vue'

const routes = [
  {
    path: '/',
    name: 'home',
    component: HomePage,
    meta: { tab: 'home', title: '首页' },
  },
  {
    path: '/translate',
    name: 'translate',
    component: () => import('../views/Translate.vue'),
    meta: { tab: 'translate', title: '翻译' },
  },
  {
    path: '/profile',
    name: 'profile',
    component: () => import('../views/Profile.vue'),
    meta: { tab: 'profile', title: '我的' },
  },
]

export default createRouter({
  history: createWebHistory(),
  routes,
})
