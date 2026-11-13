# 快速开始

目标：从 clone 到跑通第一条「检索引答」的对话，10 分钟内。

## 前置条件

| 需要 | 说明 |
|---|---|
| Windows 10+ | UI / Edge 渲染级联优先支持；Linux 可编译引擎（渲染级联需自装 Chromium） |
| Python 3.8+ | 全部插件/服务纯 stdlib，**无需 pip install 任何东西** |
| C 编译器 | 编译引擎用（MSVC 或 MinGW，见 engine.md） |
| 磁盘 ~3GB | 知识库成品 1.5GB + 微软权重 0.9GB + 代码 |

## 四步跑通

```bash
# 1. 编译引擎
cd engine && build.cmd && cd ..

# 2. 知识库成品 (二选一)
python scripts/fetch_kb.py          # 下载 77.6 万卡维基成品 (推荐)
# 或自建: 见 docs/kb.md 「自建知识库」

# 3. 微软权重 -> AIM-W
#    从 hf.co/microsoft/bitnet-b1.58-2B-4T 下载 gguf,
#    然后: python tools/gguf2aimw.py <权重.gguf> data/m2b_tern.aimw
#    详见 docs/weights.md

# 4. 起服务
python ui/server.py                 # http://127.0.0.1:8600
```

浏览器打开 `http://127.0.0.1:8600`。

## 验收清单（每项 30 秒）

| # | 动作 | 预期 |
|---|---|---|
| 1 | 对话框输入 `在吗` | 纯对话直答（不查库不联网） |
| 2 | 输入 `什么是注意力机制` | 路径=知识库引答，回答带来源卡（base 库命中） |
| 3 | 输入 `帮我搜索一下 firefox` | 路径=联网全文引答：搜索→爬取→过闸→入库→引答（1-3 分钟） |
| 4 | 输入框旁插件下拉选 `demo.time`，任意输入，点 🔌直调 | 气泡回显 JSON（`cap: demo.time`） |
| 5 | 侧栏「库存统计」 | base≈77.6万卡 / dyn 库计数 |

## 配置项

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `AIM_WORKERS` | 2（server.py 内置） | 推理引擎线程池工人数量；引擎自校准护栏会在并行不快时自动降级串行 |
| `AIM_KB_BASE` | `../kb_store` | 固定底座库（wiki 成品）路径 |
| `AIM_KB_DYN_ROOT` | `../kb_dyn_store` | 动态库（用户上传+爬虫增量）路径 |

## 常见问题

- **8600 被占用**：改 `ui/server.py` 顶部 `PORT`
- **对话很慢**：本机 CPU 推理 + 联网全链 1-3 分钟属正常；闲聊应该秒级（直答路径）
- **爬到的页都进不了库**：质量闸 fail-closed 是特性不是 bug，看「网页入库」面板每站的拒绝原因（lowqual:xxx / robots_disallow / fetch_error）
- **生成速度低**：确认 `AIM_WORKERS=2`；沙箱/云主机里 EcoQoS 节流会让速度掉一个数量级，真机不受影响
