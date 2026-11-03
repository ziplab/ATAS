
══════════════════════════════════════════
ATAS 世界状态补充 (在上面 schema 基础上增加以下字段, 其余规则不变)
══════════════════════════════════════════

本系统维护一个持久世界状态: 每个 location 是一个物理场景, 拥有共享的空间坐标 (场景记忆);
每个 shot 是一次事件观测, 按 (场景, 事件, 时间) 存入事件记忆, 后面的 shot 可以回到这个场景,
或者从新机位重拍这个历史事件。因此:

1. 顶层增加 "style": {
     "medium": "<画面实际媒介, 例: 写实电影实拍 / 2D 动画 / 3D 渲染>",
     "visual_finish": "<线条/明暗/色彩处理>",
     "genre": "<题材与年代>",
     "contract": "<一句话风格合同, 必须写明 medium 与 visual_finish, 之后所有画面都遵守>"
   }
   所有 portrait/location/anchor prompt 必须使用同一 medium。

2. location_plan 的 key 是"物理场景 ID": 同一个房间/地点在整部片里只用一个 ID, 不因景别、角度、时段、光线不同而新建。
   不同房间是不同场景。英文 slug。

3. 每个 shot 增加:
   - "location_id": "<location_plan 中的 key, 本 shot 发生的物理场景>"
   - "characters": ["<portrait id>", ...]   本 shot 在场的角色 (包括画面边缘/背景里仍在场的人)
   - "focal": ["<portrait id>", ...]        本 shot 主要拍摄的角色, 是 characters 的子集
   - "recapture_of": null 或 {
         "shot_id": "<更早的 shot id>",
         "start_s": <在那个 shot 内的起始秒>,
         "end_s": <结束秒>
     }
     只有当剧本明确要求"从另一个角度/机位再看一次之前发生过的同一事件"(回放、重看、换机位重演) 时才填,
     必须是同一 location 的更早 shot, end_s - start_s ∈ [4, 15] 且 = 本 shot duration_s。
     普通的后续动作、相似动作、回忆性描述都不是 recapture。

4. 若用户给出了固定角色列表 (见输入), portrait_plan 的 id 必须严格使用这些标签, 不得增删改名。

5. 每个 shot 的 duration_s ≥ 4 (子 agent 按 [4, 15] 拆段)。bridge 的 duration_s ∈ [4, 8]。

6. 再次强调: 镜头运动写在 visual_intent; 不要输出任何世界坐标或数值轨迹, 实际轨迹由下游根据场景点云求解。
