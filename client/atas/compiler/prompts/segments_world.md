
══════════════════════════════════════════
ATAS 补充 (每个 segment 是一次有界生成请求, 增加以下字段, 其余规则不变)
══════════════════════════════════════════

单段 duration_s ∈ [4, 15] 整数 (点云条件参考视频与生成模型上限 15 秒)。

每个 segment 额外输出:
  "shot_size": "extreme_close_up | close_up | medium_close_up | medium | medium_wide | wide | extreme_wide",
  "camera_angle": "<eye_level / low_angle / high_angle / overhead 等, 简短>",
  "camera_move_name": "locked | push_in | pull_back | orbit | arc_left | arc_right | push_in_with_arc | truck_left | truck_right | crane_up | crane_down | pan_left | pan_right | tilt_up | tilt_down | handheld | track_left | track_right | track_forward | track_back",
  "camera_brief": "<起始视角、目标构图、运动方向与幅度、结束构图; 不写坐标>",
  "focal": ["<本段主要拍摄的 portrait id>"],
  "characters": ["<本段在场的 portrait id, 包括未被聚焦但仍在场的人>"]

camera_move_name 从 visual_intent 的语义选择, 不要机械地全部 push_in; 原剧本明确的运镜必须保留方向。
pan/tilt 是原地转动, truck/crane 是平移, track 是跟随动作的平移, handheld 是轻微晃动。

输入里的 [世界状态] 是系统已经确认写入记忆的观测: 哪些场景已有空间记忆、哪些镜头被接纳。
它只用于你理解"这个地方之前拍到过什么", 不要为了迁就它改写剧情; 镜头之间的衔接关系
(新场景 / 连续 / 切机位 / 回访 / 重拍) 由系统根据世界状态自动决定, 你不需要输出。
