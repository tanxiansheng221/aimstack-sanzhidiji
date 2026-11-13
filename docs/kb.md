# 外置知识库说明书

双层库设计：**base 固定底座**（维基成品，77.6 万卡）+ **dyn 动态库**（用户上传 + 爬虫增量，随时可整包更换）。检索时两库并行查、结果合并按分排序。

## 知识库从哪来（自建）

本仓库**不分发库本体**，管线全开源，两条路：

**① 从维基 dump 建底座**（推荐起点，约 77.6 万卡）：

```bash
# 下载中文维基 dump (Wikimedia Downloads 或镜像站)
python lab/kb/build_wiki_base.py <zhwiki-dump.xml.bz2>   # 解析 -> 分块 -> 查重 -> 入库 -> FTS
```

**② 从爬虫增量**：`web.ingest` 插件（见 creeper.md）按主题搜→爬→过闸→入库，底座上叠增量——这是本栈的设计用法，知识库是活的，不是死的快照。

无论哪条路，产物都是 `kb_store/cards.jsonl` + `kb_store/kb_fts.db` 两个文件，目录可整包搬移。

## 自建 / 增量

| 文件 | 用途 |
|---|---|
| `lab/kb/build_wiki_base.py` | 从维基 dump 构建底座库（大工程，建议直接用成品） |
| `lab/kb/kb_core.py` | 核心库引擎：卡片追加（O(1)）、SHA-256 内容寻址查重闸 |
| `lab/kb/fts_index.py` | FTS5 索引封装（增量索引，只插新卡不全库重扫） |
| `lab/kb/ingest.py` | 富格式解析：txt/md/html/docx/pdf → 纯文本 |
| `lab/kb/kb_add.py`（插件） | 入库管线：load_any → 分块 → 查重（fail-closed）→ cards.jsonl → FTS |
| `lab/kb/cli.py` | 命令行直调 |

入库的规矩：**查重闸只拦重复，质量闸拦低质**（爬虫线的质量五信号在 creeper.md）。占位文件（去注释后 <30 字符）直接跳过。

## 检索

```bash
# CLI 直调
aimp_probe x --cap kb.search "线性注意力" 5

# 或经 UI「知识库检索」面板
```

`kb_search.py`（插件）实现 base+dyn 双库混合检索，合并去重按 `score` 降序。每张卡返回 `title/text/source/score/store(base|dyn)`。

## 动态库换装

`kb_dyn_store/` 整目录是自包含的（卡片 + 索引），换库 = 换目录 + 改 `AIM_KB_DYN_ROOT` 环境变量。这是「知识库外置」设计的核心：底座不动，动态层随时重开。

## 注意事项

- cards.jsonl 是追加式 NDJSON，**不要手工编辑**（查重靠 SHA-256 内容寻址，手工改会造成索引与卡片漂移）
- FTS 索引可随时重建（`purge_base_fts.py` / 重建工具），卡片本体才是真相源
- 卡片分块策略针对正文页调过：短行壳文在入库前已被质量闸砍掉
- 77.6 万卡在双核机上全量检索秒级；更大库请自行分区
