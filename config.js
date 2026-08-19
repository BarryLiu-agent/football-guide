// ⚠️ 前端直连 DeepSeek 的受限 Key（方案1：公开页面内嵌，有泄露风险）
// 用途：仅用于比赛详情弹窗内的"追问 AI"聊天。
// 请使用低额度/受限 Key，泄露后到 DeepSeek 控制台作废重建，并同步更新本文件。
window.CHAT_CONFIG = {
  API_KEY: 'sk-25a932783de240f581e7924f27bddaf6',
  BASE_URL: 'https://api.deepseek.com/v1/chat/completions',
  MODEL: 'deepseek-chat',
  MAX_HISTORY: 8,        // 保留最近 8 条消息（4 轮对话），防止 token 膨胀
  MAX_INPUT: 500         // 单次提问最大字符数
};
