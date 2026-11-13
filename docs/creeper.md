# 搜索引擎 / 爬虫说明书

自研爬虫线：多引擎搜索入口 → 并行抓取 → Edge 渲染级联 → 质量五信号闸 → 入库。全部纯 stdlib（渲染级联用系统自带 Edge，不装浏览器）。

## 文件地图

| 文件 | 职责 |
|---|---|
| `serp.py` | 四引擎 SERP 适配器（bing/百度/360/搜狗）+ 多引擎并行 + 结果块级摘要抽取 |
| `engine.py` | FetchKernel（robots 检查 + 域名限速 + 大小上限）+ fetch_generic + MediaWiki 官方 API 适配器 |
| `render.py` | Edge 渲染级联（Playwright `channel=msedge`，零浏览器下载） |
| `quality.py` | 质量五信号闸，fail-closed |
| `crawl_urls.py` | 批量抓取编排 |
| `github_adapter.py` | GitHub 官方 API 适配器（不用爬虫抓 GitHub） |

## 搜索层（serp.py）

- 三引擎**线程池并行**发车（不排队），交错合并去重
- 结果块级抽取：bing 的 `b_algo`+`b_caption` 块、百度/360/搜狗按 `<h3>` 切块，snippet 带 HTML 实体解码
- 壳页自动走渲染级联（裸抓 <8KB 或缺结果块标记 → Edge 重抓）
- 反爬壳页（搜狗 IP 级风控）诚实返回空，不硬闯

robots 实测结论（写进代码注释的）：四家引擎的搜索入口路径**全部** robots 禁抓（`/search`、`/s?`、`/s`、`/web?`）——这是行业通例。本栈把 SERP 当查询接口用（小量、低频），**正文一律从第三方网页抓，逐站过 robots**，拒绝不级联。

## 抓取层（web_ingest 插件编排）

```
SERP 结果 → 域名多样化取 topN（优质源白名单排前：MDN/维基/官方文档类）
  → 并行裸抓（4 工人线程池）
  → 串行渲染级联（残文壳页/超时站，robots 先查）
  → 质量五信号闸（fail-closed）
  → kb.add(origin=crawl) → FTS 增量索引
```

## 质量五信号闸（quality.py）

| 信号 | 拦什么 |
|---|---|
| residual_too_short | 去壳后残文 <80 字 |
| shell_ratio | 登录/关注/导航壳词占比过高 |
| too_few_sentences | 无句读结构（非散文页） |
| low_density | 信息密度低（CJK 字符多样性 / 拉丁词多样性双通道取优） |
| menu_smell | 菜单列表味（大量超短行） |

全 fail-closed：任一信号不满足即拒收并给出原因。密度信号是双通道 max——技术文档混代码/术语会稀释 CJK 占比，纯 CJK 口径会误杀，已修。

## 渲染级联（render.py）

- `render_html`（保 HTML，SERP 用）/ `render_fetch`（出正文文本）
- Playwright `channel=msedge`：用系统自带 Edge，**零浏览器下载**
- 只对「残文壳页 / 裸抓超时断连」的站级联，且先过 robots
- 已知抓不了：JS 环境检测站（如 support.mozilla.org 的完整性校验），诚实跳过

## 新增一个搜索引擎

1. 在 `serp.py` 写 `search_xxx(query, n)`，返回 `[{"title","url","snippet"}]`（记得过滤引擎自家域名页——实测会混进推广页）
2. 注册进 `ENGINES` 字典
3. 复测 robots 态度并把结论写进注释（四家现状：全禁搜索页，见上）
4. `probe_engines.py` 式探针先验后接

## 新增优质源白名单

`web_ingest.py` 的 `PRIORITY` 元组按序排优质域名（实测正文质量过硬的站），匹配者排到候选队首、同层保持 SERP 原序。
