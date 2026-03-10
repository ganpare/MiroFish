<template>
  <router-view />
</template>

<script setup>
// Vue Routerを使用してページを管理する
</script>

<style>
/* グローバルスタイルのリセット */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

#app {
  font-family: 'JetBrains Mono', 'Space Grotesk', 'Noto Sans SC', monospace;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #000000;
  background-color: #ffffff;
}

/* スクロールバーのスタイル */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: #f1f1f1;
}

::-webkit-scrollbar-thumb {
  background: #000000;
}

::-webkit-scrollbar-thumb:hover {
  background: #333333;
}

/* グローバルボタンのスタイル */
button {
  font-family: inherit;
}
</style>
