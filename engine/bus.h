/* aimp plugin bus — cap 注册表 + 调用协议 (零第三方依赖) */
#ifndef AIM_BUS_H
#define AIM_BUS_H
#include <stddef.h>

/* 初始化: 读 caps.json (NULL -> env AIM_BUS_CAPS -> plugins/caps.json)
 * rc: 0 ok / 1 读失败 / 2 无有效 cap */
int aim_bus_init(const char *caps_path);

/* 已登记 cap 数 / 按序取名 */
int aim_bus_ncaps(void);
const char *aim_bus_capname(int i);

/* JSON 字符串转义 (引号/反斜杠/控制符), out 需 >= strlen(in)*2+1 */
void aim_json_escape(const char *in, char *out, size_t cap);

/* 调一个 cap: query_json 写临时文件 -> 插件命令模板 %s 替换路径 -> _popen
 * 收 stdout 为结果 JSON。rc: 0 ok / 4 cap 未登记 / 3 spawn 失败 / 5 插件无输出
 * secs: 纯调用耗时 (秒, 不含总线初始化) */
int aim_bus_call(const char *cap, const char *query_json, char *resp,
                 size_t resp_cap, double *secs);

#endif
