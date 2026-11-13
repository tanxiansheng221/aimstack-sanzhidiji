# 插件总线说明书

约百行 C + 纯 stdlib Python 的插件系统。设计原则：**模型不感知插件，路由决定调用**。

## 协议

```
调用方 (C 引擎 / UI / 你的脚本)
  │  aim_bus_call("cap.name", '{"query":"...","topk":5}', ...)
  ▼
bus.c: 读 caps.json 注册表 → 把查询写成临时 JSON 文件 → 按模板拉起子进程
  ▼
Python 插件: argv[1]=查询文件路径 → 处理 → stdout 吐一个 JSON 对象
  ▼
调用方拿到 stdout 里的 JSON ({"cap":"cap.name", ...})
```

## 注册表 `plugins/caps.json`

```json
{
  "kb.search":  "python H:/路径/plugins/kb_search.py %s",
  "web.search": "python H:/路径/plugins/web_search.py %s",
  "kb.add":     "python H:/路径/plugins/kb_add.py %s",
  "web.ingest": "python H:/路径/plugins/web_ingest.py %s",
  "demo.time":  "python H:/路径/plugins/demo_time.py %s"
}
```

`%s` = 查询 JSON 文件路径。C 侧对 caps.json 做微扫描（`"cap" : "cmd"` 键值对），不需要完整 JSON 解析器。

## 插件契约（三条）

1. `argv[1]` 收查询文件路径，内容形如 `{"query":"...","topk":5}`
2. stdout **只输出一个 JSON 对象**（多余打印会污染解析；调试信息写 stderr 或日志文件）
3. 输出对象带 `"cap": "<自己的名字>"` 字段

最小范例就是 `plugins/demo_time.py`（20 行）。

## 加一个新插件（两步，零改 C 代码）

```python
# plugins/my_cap.py
import io, json, sys
req = json.load(io.open(sys.argv[1], encoding="utf-8"))
json.dump({"cap": "my.cap", "echo": req.get("query", "")}, sys.stdout, ensure_ascii=False)
```

```json
// caps.json 加一行
"my.cap": "python H:/路径/plugins/my_cap.py %s"
```

验证：`aimp_probe x --cap my.cap hello`，或 UI 对话框下拉选 `my.cap` 点 🔌直调。

## 内置五个插件

| 插件 | 输入 | 输出要点 |
|---|---|---|
| `kb.search` | query, topk | `results[]`：base+dyn 两库混合检索，按分排序 |
| `kb.add` | 文件路径 | `added/rejected/fts_indexed`：解析→分块→查重→FTS 增量索引 |
| `web.search` | query, topk | `results[]`：多引擎并行 SERP（含 snippet 摘要） |
| `web.ingest` | query, topn | `crawled/rejected/details[]`：并行抓取→渲染级联→质量闸→入库，见 creeper.md |
| `demo.time` | 任意 | 回显 + 时间戳，总线协议冒烟用 |

## 业务链里谁在调

对话 auto 模式（`ui/server.py` 的路由，确定性正则，不是模型）：

```
闲聊白名单命中 ──────────────► 模型直答
知识题 ──► [kb.search] 命中 ──► --ask 检索引答
查不到 ──► [web.search] ──► [web.ingest] ──► [kb.search] ──► --ask / 摘要兜底 / 诚实拒答
```

任何一步都能在 UI 的运行日志面板看到轨迹。
