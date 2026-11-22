# MarketOracle 前端

MarketOracle 的单股研究工作台包含行情、研究问答、实时多空发言、证据详情和完整研报。首页可浏览示例，提交问题后通过本地后端开始独立研究会话。

行情卡上的交换按钮切换标的；「小白 / 专业」在输入框左下角。桌面端按 `Enter` 发送、`Shift+Enter` 换行，触屏端用发送按钮。问答实时接收正文并支持 Markdown；完整研报在多空辩论和结构化汇总完成后打开，不会发送或覆盖输入框里的草稿。

## 开发

从仓库根目录打开两个 PowerShell 终端。第一个运行后端：

```powershell
& 'D:\miniforge3\envs\marketoracle\python.exe' -X utf8 -m backend.web
```

第二个运行前端：

```powershell
Set-Location frontend
npm install
npm run dev
```

开发地址：http://127.0.0.1:5173/

Vite 将 `/api` 转发到本机 8765 端口。后端从项目根目录的 `.env` 加载模型配置。选择“模拟行情和资讯”会使用模拟行情和资讯来源，生成回答仍会调用已配置的模型。网页刷新后开启新的浏览器会话。

## 构建

```powershell
npm run build
```
