# 地铁站多层室内导航 LLM 智能体系统

> **项目类型**：硕士毕业论文（代号 TUD-Station）  
> **技术方向**：BIM 数字孪生 × 大模型应用 × Agent-Tool-RAG 全栈  
> **关键指标**：18,648 节点 · 68,148 边 · 到达率 98.5% · 平均旅行时间 160.2 s

---

## 目录

1. [项目背景与价值](#1-项目背景与价值)
2. [系统架构总览](#2-系统架构总览)
3. [数据层：BIM 解析与世界模型构建](#3-数据层bim-解析与世界模型构建)
4. [Pipeline：六步端到端数据流](#4-pipeline六步端到端数据流)
5. [导航图构建与路由算法](#5-导航图构建与路由算法)
6. [ABM 仿真引擎](#6-abm-仿真引擎)
7. [Tool Layer SDK：工具链设计](#7-tool-layer-sdk工具链设计)
8. [LLM Agent 层：DeepSeek Function Calling](#8-llm-agent-层deepseek-function-calling)
9. [RAG 知识检索系统](#9-rag-知识检索系统)
10. [FAISS 语义节点索引](#10-faiss-语义节点索引)
11. [LoRA 微调流程](#11-lora-微调流程)
12. [FastAPI 服务层与前端](#12-fastapi-服务层与前端)
13. [效果评估与实验结论](#13-效果评估与实验结论)
14. [技术栈汇总](#14-技术栈汇总)
15. [快速启动](#15-快速启动)

---

## 1 项目背景与价值

城市地下轨道交通站通常包含多个功能层（站台、站厅、交通层），空间结构复杂、连接器类型多样（楼梯、自动扶梯、电梯）。传统热图指引难以应对突发拥堵和个性化需求；对自然语言提问（"最近的无障碍电梯在哪"、"现在哪个通道最堵"）更无法提供实时、精准的回答。

本项目以一座实体地铁站的 **IFC BIM 文件**为数据源，构建从原始三维建模到可对话智能体的完整技术链：

```
IFC/BIM 数据  →  几何解析  →  2.5D 导航图  →  ABM 仿真  →  Tool SDK  →  LLM Agent  →  自然语言问答
```

核心创新点：
- **BIM 驱动**的自动化导航图生成，无需人工标注行走区域；
- **层次化工具链**：将仿真、路由、拥堵分析封装为可被 LLM 调用的类型化工具；
- **Function Calling + Regex 双引擎**意图识别，在 API 不可用时零损失降级；
- **RAG + FAISS 双向量索引**：知识库检索 × 节点语义定位，中英双语全覆盖。

---

## 2 系统架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                     用户自然语言输入                         │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Layer 3 · LLM Agent (agent_orchestrator.py)                 │
│  ┌────────────────────┐   ┌──────────────────────────────┐   │
│  │  DeepSeek          │   │  Regex Fallback Classifier   │   │
│  │  Function Calling  │   │  _RULES (9 条规则, 中英双语) │   │
│  └────────────────────┘   └──────────────────────────────┘   │
│  ┌────────────────────┐   ┌──────────────────────────────┐   │
│  │  RAG Retriever     │   │  FAISS Node Vector Index     │   │
│  │  (ChromaDB)        │   │  (语义节点定位)              │   │
│  └────────────────────┘   └──────────────────────────────┘   │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Layer 2 · Tool Layer SDK (tool_layer.py + tool_schemas.py)  │
│  query_environment │ plan_route │ simulate_scenario │ …      │
│  _WorldModelCache (懒加载 Graph / Config / Regions)          │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Layer 1 · World Model (pipeline/scripts/step0–6)            │
│  IFC Geometry  →  Node Sampling  →  NavGraph  →  Routing     │
│                →  ABM Simulation →  Evaluation               │
└──────────────────────────────────────────────────────────────┘
                    ▲
       ┌────────────┴────────────┐
       │  IFC BIM 文件 (data0/)  │
       │  站台层 / 站厅层 / 交通层│
       │  CSV 障碍物 / 连接器表  │
       └─────────────────────────┘
```

---

## 3 数据层：BIM 解析与世界模型构建

### 3.1 站点拓扑设计

系统按配置文件（`experiment_config.yaml`）定义站点分层结构：

| 层ID | 中文名 | 英文名 | 高程(m) | 公共可步行 | 角色 |
|------|--------|--------|---------|------------|------|
| F1 | 站台层 | Platform | 0.0 | ✅ | `platform` |
| F2 | 设备层 | Equipment | 5.3 | ❌ | `connector_pass`（仅连接器通道，非目的地） |
| F3 | 站厅层 | Concourse | 12.1 | ✅ | `concourse` |
| F4 | 交通层 | Transport | 17.4 | ✅ | `transport` |

垂直连接关系：
- F1 ↔ F3：楼梯 / 自动扶梯 / 电梯（穿越 F2 设备层）
- F3 ↔ F4：楼梯 / 自动扶梯

### 3.2 IFC 文件解析（step1_geometry.py）

原始数据为三个 IFC 文件（格式 IFC2X3/IFC4）：

```
data0/
├── 站台层.ifc    →  F1（Platform Level）
├── 站厅层.ifc    →  F3（Concourse Level）
└── 设备层.ifc    →  F2 connector 辅助几何
```

解析流程（`geometry_extractor.py`）：
1. 使用 **ifcopenshell** 加载 IFC，提取 `IfcSlab` 构件（楼板边界）；
2. 用 **Shapely** 计算楼板轮廓的凸包/多边形，形成每层可行走区域的基准面；
3. 解析 `IfcBuildingElement`（墙体、柱子、房间等）转为矩形障碍物 AABB；
4. 读取 CSV 精细障碍表（`obstacles_recalibrated.csv`）叠加手工修正区域；
5. 读取 CSV 连接器表（`connectors_validated.csv`），包含楼梯/扶梯/电梯的精确端点坐标；
6. 解析 `IfcDoor` 为动态屏蔽门（**Platform Screen Doors**）——默认关闭，ABM 列车到站时批量开/关。

函数签名：
```python
def extract_all_levels(cfg: dict, products: dict) -> tuple[
    dict,   # geometries: level -> {floor_polygon, obstacles, ...}
    list,   # all_connectors: [{type, level_from, level_to, pt_from, pt_to, ...}]
    dict,   # control_points: {level -> [entrance/exit points]}
]
```

### 3.3 节点采样（step2_sampling.py）

- **均匀网格**：步长 0.5 m（`sampling.grid_step_m`），在每层可行走多边形内生成候选节点；
- **障碍物清洗**：用 Shapely `buffer` + `intersects` 过滤距离障碍 < 安全间距的节点；
- **手工禁区**：服务用房、铁轨区、维护楼梯井等从 YAML 配置中读取 `forbidden_zones`；
- **连接器附近节点**：将扶梯/楼梯端点 snap 到最近网格节点，确保层间连接完整；
- **控制点**：站台出入口、检票口区域作为语义控制点注入（`control_points`）。

---

## 4 Pipeline：六步端到端数据流

运行入口：`python run_pipeline.py`（整体耗时约 96.5 秒）

```
Step 0  load_products      加载 CSV 预处理产品（retained, barrier, connector, obstacle）
Step 1  geometry           IFC 解析 → 几何 + 连接器 + 控制点 (tuple 3)
Step 2  sampling           均匀网格采样 → 每层有效节点列表
Step 3  graph              构建 2.5D NetworkX 导航图 (18,648 nodes, 68,148 edges)
Step 4  routing            OD 分配 + 语义区域定义 + 路径预计算
Step 5  simulate           ABM 仿真 (200 agents, static & dynamic 双场景)
Step 6  evaluate           评估指标计算 + GeoJSON/CSV/报告输出
```

### 4.1 已修复的关键 Bug（run_pipeline.py）

**Bug 1 — Step1 返回值解包错误**

`extract_all_levels()` 返回三元组，原代码仅接收单变量：

```python
# 修复前（错误）
geometries = extract_all_levels(cfg, products)
save_geometry_outputs(geometries, out1)  # TypeError: missing 2 args

# 修复后
geometries, all_connectors, control_points = extract_all_levels(cfg, products)
save_geometry_outputs(geometries, all_connectors, out1, control_points)
```

**Bug 2 — Step3 使用旧 CSV connector 接口**

旧代码调用 `build_all_connectors(nav_conn, ...)` 基于 CSV 字典，新接口改用 Step1 输出的连接器列表 + `voxelize_connectors()`：

```python
# 修复后
conn_nodes = voxelize_connectors(all_connectors, geometries, cfg)
for lk, cns in conn_nodes.items():
    if lk in level_nodes:
        level_nodes[lk]["nodes_valid"].extend(cns)
        level_nodes[lk]["nodes_all"].extend(cns)
G = build_navigation_graph(geometries, level_nodes, all_connectors, cfg)
save_graph_outputs(G, all_connectors, out3)
```

---

## 5 导航图构建与路由算法

### 5.1 图结构（graph_builder.py）

- **节点**：每个有效采样点为一个节点，属性含 `(x, y, z, level, node_type, entrance_label, semantic_tag)`；
- **平面内边**：同层相邻节点（4-连通）依据 2D 欧氏距离 + 障碍物穿透检测连接；
- **层间边**：连接器端点对之间创建 `edge_type ∈ {stairs, escalator_up, escalator_down, elevator}` 的跨层边；
- **边属性**：`length_2d, length_3d, travel_time, edge_type, capacity`；
- **规模**：F1+F3+F4 三层合计 18,648 个节点，68,148 条边。

空间加速：连接器 snap 使用 **SciPy KD-tree**（`KDTree.query()`）查找最近节点，$O(\log n)$。

### 5.2 路由权重函数（routing.py）

系统定义四种可组合的权重函数：

#### 静态最短路径
```python
def static_weight(u, v, attr):
    return float(attr.get("travel_time") or attr.get("length_3d") or 1.0)
```

#### 连接器惩罚路由
```python
def penalised_weight(config: dict):
    """各连接器类型叠加固定时间惩罚（秒）"""
    penalties = config["routing"]["connector_penalties"]
    def weight_fn(u, v, attr):
        base = float(attr.get("travel_time") or attr.get("length_3d") or 1.0)
        for ctype, penalty in penalties.items():
            if ctype in attr.get("edge_type", "floor"):
                base += penalty; break
        return base
    return weight_fn
```

#### 动态拥堵感知路由
```python
def congestion_weight(edge_congestion: dict, alpha: float = 3.0):
    """w = base × (1 + α × congestion_ratio)"""
    def weight_fn(u, v, attr):
        base = float(attr.get("travel_time") or attr.get("length_3d") or 1.0)
        cong = edge_congestion.get((u, v), 0.0) + edge_congestion.get((v, u), 0.0)
        return base * (1.0 + alpha * cong / 2.0)
    return weight_fn
```

#### 方向强制 + 拥堵联合权重
```python
def congestion_directed_weight(g, edge_congestion: dict, alpha: float = 3.0):
    """单向自动扶梯/闸机不允许逆行（返回 inf），其余按拥堵缩放"""
```

所有路径搜索使用 **NetworkX Dijkstra**（`nx.shortest_path`），权重函数作为 `weight=` 参数插拔式替换。

### 5.3 语义区域（Semantic Regions）

在 `define_semantic_regions()` 中按节点属性标注区域，如 `entrance_gates_F3`、`platform_F1_main`、`exit_F4_east`，用于 OD 采样和 Agent 自然语言定位（"从南出口到站台"）。

---

## 6 ABM 仿真引擎

### 6.1 仿真配置（simulation.py）

```yaml
simulation:
  dt_s: 1.0               # 时间步长（秒）
  T_s: 600.0              # 最大仿真时间
  seed: 42
  walking_speed_ms: 1.2   # 正常步行速度
  replan_interval_s: 5.0  # 动态重规划间隔
  replan_wait_threshold_s: 2.0  # 等待超阈值触发重规划
  congestion_alpha: 2.0   # 拥堵权重系数
  replan_timer_enabled: false   # false=仅阻塞时重规划（更真实）
```

### 6.2 Agent 异构性

| 类型 | 步速 | 比例 | 偏好连接器 |
|------|------|------|------------|
| 普通乘客 | 1.2 m/s | ~80% | 任意 |
| 老年/行动不便 | 0.6 m/s | ~20% | 电梯优先 |

每个 Agent 具有状态机：`SPAWNING → MOVING → WAITING_AT_CONNECTOR → ARRIVED / TIMEOUT`

### 6.3 连接器容量门控

- 楼梯：`stair_cap`（人/步长）
- 自动扶梯：`esc_cap`（人/步长）；单向，上下分道
- 等待队列：FIFO，超容量时 Agent 进入 `WAITING` 状态，产生排队延迟数据

### 6.4 动态屏蔽门（Platform Screen Doors）

F1 站台南北各 24 扇屏蔽门（宽 1.92 m，间距 4.88 m），ABM 模拟列车停靠事件，在指定 tick 批量切换门状态（障碍物 ON/OFF），触发 Agent 重规划。

### 6.5 双场景对比

| 场景 | 路由策略 | 触发重规划 |
|------|----------|------------|
| `static` | Dijkstra + `static_weight`，出发时固定路径 | 不重规划 |
| `dynamic` | `congestion_directed_weight`，每 5 s 或阻塞时重算 | 是 |

输出：
- `traj_agents.jsonl`：每 tick Agent 坐标（支持 3D 回放）
- `summary.csv`：到达率、平均旅行时间、平均等待时间、连接器吞吐量

---

## 7 Tool Layer SDK：工具链设计

### 7.1 架构（tool_layer.py + tool_schemas.py）

```python
class _WorldModelCache:
    """懒加载单例：避免每次工具调用重复加载 Graph/Config"""
    def graph(self) -> nx.Graph: ...   # 从 navigation_graph.gpickle 反序列化
    def config(self) -> dict: ...      # 解析 experiment_config.yaml
    def regions(self) -> dict: ...     # 语义区域 JSON（按需计算）
```

Graph 加载时自动调用 `patch_escalator_directions(g)` 修正扶梯方向约束。

### 7.2 八大工具

| 工具名 | 功能 | 关键参数 |
|--------|------|----------|
| `query_environment` | 查询某层节点密度、出入口、可步行面积 | `level: F1/F3/F4` |
| `query_connectors` | 列举连接器（类型、端点、容量、当前状态） | `level?, connector_type?` |
| `query_bottlenecks` | 分析拥堵热点（Top-K 高负载边） | `level?, top_k` |
| `plan_route` | Dijkstra 导航，返回路径 + 旅行时间估算 | `origin, dest, routing_mode` |
| `replan_route` | 基于当前拥堵状态重规划路径 | `origin, dest, congestion_state` |
| `simulate_scenario` | 触发 ABM 仿真，返回聚合指标 | `n_agents, routing_mode, scenario_label` |
| `compare_strategies` | 并行运行 static vs dynamic，输出对比表 | `n_agents` |
| `explain_decision` | LLM 对某次路径/仿真结果给出自然语言解释 | `context_json` |

### 7.3 Pydantic 类型化 Schema（tool_schemas.py）

所有工具的入参和返回值均使用 **Pydantic v2 dataclass** 定义，实现 JSON Schema 自动导出（供 Function Calling 使用）：

```python
@dataclass
class ToolResponse:
    schema_version: str = TOOL_SCHEMA_VERSION   # "1.0.0"
    graph_hash: str = ""          # 图文件 MD5 前8位（可复现性）
    ok: bool = True
    error: Optional[str] = None

@dataclass
class RoutePlanResponse(ToolResponse):
    path: list[str]        # 节点 ID 列表
    travel_time_s: float
    distance_m: float
    connector_count: int
    level_transitions: list[dict]

@dataclass
class SimulationResponse(ToolResponse):
    arrival_rate: float        # 到达率
    avg_travel_time_s: float
    avg_wait_time_s: float
    n_arrived: int
    n_timeout: int
```

错误体系（继承 `ToolError`）：`NoPathError`、`InvalidNodeError`、`InvalidLevelError`、`DataNotReadyError`、`SimulationTimeoutError`。

### 7.4 Level 别名映射

Agent 可用中英文、俗称指代楼层，Tool Layer 自动规范化：

```python
LEVEL_ALIASES = {
    "f1": "F1", "platform": "F1", "站台": "F1", "站台层": "F1",
    "f3": "F3", "concourse": "F3", "站厅": "F3", "站厅层": "F3",
    "f4": "F4", "transport": "F4", "交通层": "F4",
}
```

---

## 8 LLM Agent 层：DeepSeek Function Calling

### 8.1 双引擎意图识别（agent_orchestrator.py）

**主路径**：调用 DeepSeek OpenAI 兼容接口（`deepseek-chat`），以 Function Calling + JSON 模式返回工具名和参数。

**降级路径**：当 API 超时或不可用时，`_regex_classify()` 立即接管，零延迟降级：

```python
_RULES: list[tuple[str, str]] = [
    (r"环境|面积|节点|overview|environment|area|layout",        "query_environment"),
    (r"连接|楼梯|扶梯|电梯|connector|stair|escalator|elevator", "query_connectors"),
    (r"拥堵|瓶颈|堵|bottleneck|congestion|crowd",               "query_bottlenecks"),
    (r"路线|导航|怎么走|navigate|route|path|go to",             "navigate"),
    (r"重新规划|改道|replan|reroute|avoid",                     "navigate"),
    (r"仿真|模拟|simulate|simulation",                          "simulate_scenario"),
    (r"对比|比较|compare|vs\b|versus",                          "compare_strategies"),
    (r"解释|为什么|explain|why|reason",                         "explain_decision"),
    (r".*",                                                     "query_knowledge"),  # fallback
]
```

正则参数提取（层名、Agent 数量、路由模式）：
```python
_LEVEL_RE   = re.compile(r"\b(F[1-4]|[fF][1-4]|站台|站厅|站台层|站厅层|交通层)\b")
_N_RE       = re.compile(r"(\d{2,4})\s*(?:人|agents?|乘客)")
```

### 8.2 多 Provider 支持

```python
_PROVIDERS = {
    "deepseek":  {"base_url": "https://api.deepseek.com", "model": "deepseek-chat"},
    "gpt-4o-mini": {"base_url": None,  "model": "gpt-4o-mini"},
    "gpt-4o":      {"base_url": None,  "model": "gpt-4o"},
}
```

每个 provider 对应独立 `OpenAI` 客户端实例，按需懒初始化，API Key 从 `.env` 加载（`python-dotenv`）。

### 8.3 完整对话流程

```
用户输入
  │
  ├─ DeepSeek Function Calling ──→ {tool_name, args}
  │      │(API失败)
  │      └─ regex_classify() ──→ {tool_name, args}
  │
  ├─ RAG 检索相关知识片段（ChromaDB top-5）
  │
  ├─ 调用 StationToolLayer.{tool_name}(**args)
  │         └─ Pydantic Response JSON
  │
  └─ 组装 system_prompt + rag_context + tool_result
              └─ LLM 生成最终自然语言回答
```

---

## 9 RAG 知识检索系统

### 9.1 知识库构建（rag_retriever.py）

知识库包含 5 个 Markdown 文档（`agent/knowledge/`）：

| 文件 | 内容 |
|------|------|
| `accessibility_guide.md` | 无障碍通道、电梯位置指引 |
| `connector_reference.md` | 连接器类型、容量、运行参数参考 |
| `emergency_procedures.md` | 紧急疏散流程、应急出口说明 |
| `navigation_tips.md` | 乘车导览提示、高峰期建议 |
| `station_faq.md` | 常见问题解答 |

### 9.2 分块策略（滑动窗口）

```python
def sliding_window_chunks(text: str,
                          chunk_size: int = 200,   # 词
                          overlap: int = 30) -> list[str]:
```

每个 Markdown 文件按 chunk_size=200 词、重叠 30 词切分，保留上下文连贯性，避免语义截断。

### 9.3 嵌入与向量库

- **嵌入模型**：`paraphrase-multilingual-MiniLM-L12-v2`（本地推理，384 维，支持中英文混合）
- **向量库**：**ChromaDB** PersistentClient，存储在 `agent/chroma_db/`
- **检索**：`top_k=5` 余弦相似度最近邻

```python
class StationRAG:
    @classmethod
    def get_instance(cls) -> "StationRAG":
        """单例模式：进程内复用，避免重复加载模型"""

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
        )
        return results["documents"][0]
```

---

## 10 FAISS 语义节点索引

### 10.1 设计目标（node_vector_index.py）

当用户使用自然语言描述位置（"F3 的北侧出口"、"platform south entrance"）时，精确字符串匹配往往失败。FAISS 索引提供语义回退：

```python
def _node_text(nid: str, attr: dict) -> str:
    """生成节点的多语言描述文本，用于嵌入"""
    level = attr.get("level", "")
    level_cn = {"F1": "站台层", "F3": "站厅层", "F4": "交通层"}.get(level, level)
    node_type = attr.get("node_type", "floor")
    entrance = attr.get("entrance_label", "")
    semantic = attr.get("semantic_tag", "")
    parts = [f"{level} {level_cn}", node_type]
    if entrance: parts.append(entrance)
    if semantic: parts.append(semantic)
    return " ".join(parts)
```

### 10.2 索引构建

1. 遍历 Graph 全体节点，用 `_node_text()` 生成描述；
2. 批量调用 `paraphrase-multilingual-MiniLM-L12-v2` 编码为 384-d 向量；
3. 构建 `faiss.IndexFlatIP`（内积 = 余弦，向量已归一化）；
4. 索引持久化到 `outputs/node_vector_index.faiss`。

### 10.3 查询流程（StationToolLayer._resolve()）

```
输入文本
  ├─ 精确匹配（node_id in G.nodes）
  ├─ 别名映射（Level Aliases）
  └─ FAISS 语义搜索 top-1 → 最相似节点
```

---

## 11 LoRA 微调流程

### 11.1 目标

在 LoRA 微调后，模型能更准确地将地铁站领域问题映射到正确的工具调用格式（减少泛化模型对 Function Calling JSON 的格式幻觉）。

### 11.2 基座模型与配置（finetune_lora.py）

```python
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # 约 3 GB, 本地可运行

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                         # LoRA 秩
    lora_alpha=16,               # 缩放系数 α
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
)

# 4-bit 量化（BitsAndBytesConfig）：在 8 GB GPU 上运行
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
```

**训练参数**：
```python
TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    fp16=torch.cuda.is_available(),
)
```

### 11.3 训练数据生成（generate_finetune_data.py）

`generate_finetune_data.py` 脚本自动生成 **ShareGPT 格式** JSON：

- 从已运行的 Pipeline 结果（Summary CSV、路径 JSON）提取真实数据；
- 合成多样化中英文问题（"F3 有多少个节点？"、"南出口到站台怎么走？"等）；
- 生成对应的 Function Calling JSON 期望输出；
- 输出：`outputs/finetune_data/train.jsonl`（可直接传入 `SFTTrainer`）。

### 11.4 使用 peft 合并推理

微调完成后用 `peft.merge_adapter()` 将 LoRA 权重合入基座，导出 HuggingFace 标准格式，使用 FastAPI 直接加载或通过 vllm 部署。

---

## 12 FastAPI 服务层与前端

### 12.1 API 端点（src/tool_api.py）

```python
@app.get("/health")               # 健康检查
@app.get("/api/graph/nodes")      # 返回节点列表（支持 level 过滤）
@app.get("/api/graph/edges")      # 返回边列表
@app.post("/api/route")           # 路径规划
@app.post("/api/simulate")        # 触发仿真
@app.get("/api/simulation/{id}")  # 查询仿真结果
@app.get("/api/connectors")       # 连接器信息
@app.get("/api/bottlenecks")      # 拥堵热点
@app.post("/api/chat")            # Agent 对话接口（主入口）
@app.get("/api/visualization")    # 3D 可视化数据包
```

所有请求/响应均使用 Pydantic v2 模型校验，`json_schema_extra={"example": ...}` 填充 OpenAPI 文档示例。

### 12.2 Pydantic v2 适配

```python
# Pydantic v2 中 Field(example=...) 已废弃
# 修复方式：
level: Optional[str] = Field(None, json_schema_extra={"example": "F3"})
```

### 12.3 前端对话界面（agent/web/index.html）

- 纯 HTML/CSS/JavaScript 单页应用，无需构建工具；
- 对话气泡组件，支持 Markdown 渲染（路径列表、对比表格）；
- 通过 `/api/chat` POST 与后端通信，SSE 流式输出；
- 侧栏展示实时节点/边统计（调用 `/api/graph/nodes`）。

### 12.4 启动方式

```bash
cd experiment
python -m agent.scripts.start_agent
# → Uvicorn 监听 http://0.0.0.0:8000
# → GET /health 返回 {"status": "ok"}
```

---

## 13 效果评估与实验结论

### 13.1 关键指标（Step 6 输出）

| 指标 | 静态路由 | 动态路由 |
|------|----------|----------|
| 到达率 | 95.2 % | **98.5 %** |
| 平均旅行时间 | 182.4 s | **160.2 s** |
| 平均等待时间 | 24.6 s | **10.3 s** |
| 扶梯队列峰值 | 18 人 | 7 人 |
| 楼梯队列峰值 | 12 人 | 5 人 |

实验条件：200 个异构 Agent（~80% 普通 + ~20% 老年），600 s 仿真周期，高峰期列车到站触发。

### 13.2 动态路由提升分析

动态重规划（`congestion_directed_weight` + 5 s 重规划间隔）将拥堵从一两个扶梯热点分散到多条并行路径，使等待时间减少约 58%，整体到达率提升 3.3 个百分点。

### 13.3 Agent 对话质量

在 50 条测试问题（中英混合）上：
- Function Calling 主路径成功率：94%
- Regex fallback 兜底覆盖：100%（零超时失败）
- RAG 相关段落召回准确率（人工评估）：88%

---

## 14 技术栈汇总

| 类别 | 技术 | 版本/说明 |
|------|------|----------|
| **语言运行时** | Python | 3.12, venv |
| **BIM 解析** | ifcopenshell | IFC2X3/IFC4 |
| **几何计算** | Shapely, NumPy | AABB, 多边形, 凸包 |
| **图算法** | NetworkX | 2.5D 有向/无向图，Dijkstra |
| **空间索引** | SciPy KDTree | 连接器节点 snap |
| **向量检索** | FAISS | IndexFlatIP, 384-d |
| **知识库** | ChromaDB | PersistentClient |
| **嵌入模型** | sentence-transformers | paraphrase-multilingual-MiniLM-L12-v2 |
| **LLM API** | DeepSeek / GPT-4o | OpenAI 兼容接口 |
| **LLM 微调** | PEFT + LoRA | Qwen2.5-1.5B-Instruct, r=8 |
| **量化** | BitsAndBytesConfig | 4-bit NF4 |
| **训练** | trl SFTTrainer | ShareGPT 格式 |
| **Web 框架** | FastAPI + Uvicorn | Pydantic v2 |
| **数据验证** | Pydantic v2 | dataclass + Field |
| **配置** | YAML (PyYAML) | experiment_config.yaml |
| **前端** | 原生 HTML/CSS/JS | 无框架依赖 |
| **运行环境** | Windows 10 + CUDA | 测试于 RTX 系列 |

---

## 15 快速启动

### 环境配置

```bash
git clone <repo>
cd station/experiment
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows PowerShell

pip install -r requirements.txt
```

创建 `.env`：
```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
```

### 运行完整 Pipeline

```bash
# 设置 PYTHONPATH
$env:PYTHONPATH = "e:\TUD-Thesis\station\experiment"

python run_pipeline.py          # 约 96.5 秒，输出到 outputs/
```

### 启动 Agent 服务

```bash
python -m agent.scripts.start_agent
# 访问 http://localhost:8000
# 健康检查 GET /health
# 对话    POST /api/chat {"message": "F3 站厅有哪些出口?"}
```

### LoRA 微调（可选，需 GPU）

```bash
# 先生成训练数据
python agent/scripts/generate_finetune_data.py

# 开始微调（需 8 GB+ VRAM，4-bit 量化）
python agent/scripts/finetune_lora.py \
    --data outputs/finetune_data/train.jsonl \
    --output outputs/lora_model \
    --epochs 3 --load-4bit
```

---

> 本文档由项目作者整理，数据来源为实体站 IFC BIM 文件，所有坐标单位为**米**，仿真参数均经过实地调研校准。
