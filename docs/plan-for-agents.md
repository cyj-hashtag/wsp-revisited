# wsp-revisited：给编码代理的项目计划

> **读者**：在本仓库里与用户合作的 LLM 代理（Claude Code 等）。
> **动手前完整读完本文件。** 它是项目的单一事实来源。用户的指示优先于本文件；两者冲突、或你想偏离计划时，先问用户。
> 每完成一项任务，更新第 11 节的进度表并提交。
> 起草：2026-11-03，云端 Claude 会话。第 2 节的事实均已在原始来源中核对过，引用时请保留出处。

---

## 0. 项目一句话

检验 "Sparks of Weight-Space Programming"（Takehi, Teoh, Zhang, Khattab，MIT CSAIL，2026-11-02 博客）中**实验 8** 的结论：GPT-6 Astra 只改 Qwen3.5-0.8B 0.016% 的权重，就把 MATH-500 从 7.9% 提到 48.1%。

**我们的假设**：

1. 增益主要来自修复"思考停不下来"（终止失败），而不是数学能力本身；
2. 原文的基线设置让 base 处于最差状态，在官方推荐的解码设置下，base 没有那么差；
3. 不需要前沿 LLM，一个自动的、经典的方向消融流程就能找到同类编辑。

**交付物**：

- 两天内：GitHub 公开代码与结果，Zenodo 拿 DOI；arXiv 等背书到位后发布。
- 之后 2–3 个月：扩展成顶会版本（目标 ICML 2027）。

**暂定题目**：*Termination, Not Reasoning: Revisiting LLM-Driven Weight Editing of Qwen3.5-0.8B*

---

## 1. 角色与硬规则

**用户**：数学系大三本科生，唯一作者，对所有决定和论文内容负责。

- 有 Claude、GPT、DeepSeek 的 API。
- 笔记本 GPU（AMD 780M，4GB）不用于实验；实验在 AutoDL 租的 GPU 上跑。
- 交流用中文；论文用英文。

**代理**：写代码、做分析、起草文字与推导、查相关工作、主动挑刺。

**硬规则**：

1. **不编造任何数字、引用或结果。** 论文里每个数字都必须能追溯到 `results/` 下的一个文件，以及生成它的命令。没跑出来的写"TBD"。
2. **引用只用亲自打开核实过的文献**，并给出 arXiv ID 或 URL。拿不准的标 `TODO(verify)`，交给用户核实。
3. **预注册**：`prereg/criteria.md` 必须在 RQ1 正式运行之前提交并推送。之后不得修改，只能在文件末尾追加带日期的 "Amendment"，并写明原因。
4. **如实报告。** 结果不支持假设也照写。不挑配置、不挑种子。所有跑过的配置都进附录。
5. **论文文字最终由用户改写。** 起草时用朴素简洁的英文。致谢中如实披露 AI 辅助（arXiv CS 要求作者对全部内容负责）。
6. **对原作者公正。** 他们公开了全部日志，并在博客的 Gemma 一节自己写过"修循环 ≠ 变得更会数学"。论文定位是补充和推进，不是打假。对外发布前先给原作者发邮件。
7. **Git**：
   - 不直接推 `main`，在分支上工作；推送前跑相关测试。
   - 不提交模型权重或大于约 50MB 的生成文件，用 `.gitignore` 排除。大文件留在 GPU 机器上，或打包传到 Zenodo / Hugging Face。
8. **花钱的操作要先问。** 租卡、加卡、大规模 API 调用，都先给出估算并征得用户同意。

---

## 2. 已核实的背景事实

**来源缩写**：

- **[A]** 博客：<https://rikiyat.github.io/blog/weight-space-programming/>
- **[B]** 作者日志仓库：<https://github.com/RikiyaT/lm_understand-logs>
  - 脚本是 MIT 许可，日志和数据是 CC BY 4.0。
  - 实验 8 在 `exp8_math/`。克隆时加 `GIT_LFS_SKIP_SMUDGE=1`。
- **[C]** Qwen3.5-0.8B model card：<https://huggingface.co/Qwen/Qwen3.5-0.8B>（README.md）
- **[D]** 编辑后的模型及第三方复现：<https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming>

### 2.1 实验 8 的设置（[A]，[B] `exp8_math/README.md`）

- **编辑者**：GPT-6 Astra，通过 Codex CLI 调用，effort 设为 max。
- **可用工具**：
  - 一个 Python 内核，可以读写权重，只有 CPU，最多运行 300 次代码；
  - 禁止梯度下降和任何训练；
  - forward 预算为 10、25 或 50 次（三次独立运行）。
- **一次 forward 是以下两者之一**：
  - 生成：最多 512 道练习题；
  - teacher-forced 读取：最多 1024 段文本，可以拿到 hidden states、MLP 激活和 attention。
- **练习题**：GSM8K train 7,469 道，MATH train 7,275 道，见 `exp8_math/practice/`，每行一个 JSON。
- **评测协议 `think35`**（[B] `code/evaluate.py`、`code/profiles.py`）：
  - 用 vLLM 推理，chat template 设置 `enable_thinking=True`；
  - user 消息 = 题目 + `"\n"` + `"Please reason step by step, and put your final answer within \boxed{}."`；
  - 采样：temperature 0.6，top_p 0.95，top_k 20，presence_penalty 1.5，max_tokens 32768，seed 0；
  - 判分：取最后一个 `\boxed{}`，用 math-verify 0.9.0 判断（`code/mathqa.py`）。
- **每题样本数**：MATH-500 4；GSM8K 2；IFEval 2（strict prompt-level，判的是 `</think>` 之后的回答）；AMC23 16；Minerva、Olympiad、GPQA 各 4；MMLU-Pro（随机 1k 题）1。
- **注意**：作者的 `evaluate.py` 依赖一个未公开的私有包 `weightenv`（mathqa、ifevalg、mcqa 都在里面）。
  - 能直接复用的只有 `code/mathqa.py`、`apply_edit.py`、`st_numpy.py`。
  - IFEval 需要换成官方实现，例如 google-research 的 `instruction_following_eval`，或 lm-evaluation-harness 的 ifeval；口径要对齐：strict prompt-level，只判 `</think>` 之后的文本。

### 2.2 原文结果（[B] `exp8_math/eval/*/merged.json`）

| 模型 | MATH-500 | GSM8K | IFEval | MATH-500 截断率 | MATH-500 平均 token |
|---|---|---|---|---|---|
| base | 7.9 | 13.8 | 21.0 | 93.0% | 30,868 |
| p10c（10 次 forward） | 12.3 | 25.7 | 49.0 | 87.2% | 30,084 |
| p25c（25 次 forward） | 14.1 | 25.1 | 39.9 | 86.4% | 29,659 |
| p50c（50 次 forward） | 48.1 | 67.1 | 62.3 | 33.1% | 22,037 |

- base 在其他 benchmark 上的截断率：GSM8K 87.5%，IFEval 74.9%。p50c 对应为 7.2% 和 2.2%。
- **第三方复现**（[D]）：
  - p50c 在 MATH-500（每题 4 个样本）上得 **51.0%**，截断率 31.6%；
  - 共 4340 万 token，H200 上跑了 51 分钟，约 1.4 万 token/s；
  - 版本：vLLM 0.33.0，transformers 5.20.2，math-verify 0.9.0；
  - 用的是完整的多模态 checkpoint，语言模型部分的张量与作者用的纯文本版逐一哈希一致。
  - base 只做了 25 题冒烟测试：5.0%，截断率 95%。

### 2.3 p50c 的编辑内容（[B] `runs/p50c/work_notes/final_report.md`、`runs/p50c/edit/manifest.json`）

- 只改了两个张量：`model.language_model.layers.{15,19}.mlp.down_proj.weight`，形状 1024×3584，每层改 64 列。
- **公式**：W = W₀ − 6·u (m ⊙ (uᵀW₀))
  - u 是单位向量（"repetition direction"），在第 16 和第 20 层的残差输出处，用 4 对匹配的"正常 / 重复"上下文求得；
  - m 是被选中 64 列的 0/1 掩码。
- **α = 6**：被选列在 u 方向上的分量变成原来的 −5 倍。博客说的"减去该方向"低估了改动幅度。
- 相对改变量（Frobenius 范数）：第 15 层 10.6%，第 19 层 8.1%。按 bf16 计实际改变了 130,188 个元素。
- **重建方法**：
  - 命令：`python exp8_math/code/apply_edit.py <base model.safetensors> exp8_math/runs/p50c/edit <out.safetensors>`
  - 脚本会逐张量做 sha256 校验；
  - base 的版本为 Qwen/Qwen3.5-0.8B @ `2fc06364715b967f1860aea9cf38778875588b17`。
- **另外两次运行的编辑完全不同**：
  - p10c：在第 8–22 层的 30 个输出矩阵上做 W' = W − 1.5·u(uᵀW)；
  - p25c：关掉第 15 层第 7 个 attention head。

### 2.4 Astra 在练习集上的自查（[B] `runs/p50c/work_notes/final_report.md`）

1,152 道独立练习题，配对比较，每题 1 个样本，上限 32k token：

- 准确率从 10.6% 升到 60.4%；579 题从错变对，5 题从对变错。**579 题里有 576 题原本是被截断的。**
- 原始模型：1038 题截断，114 题正常结束，其中 **107 题答对（94%）**。
- 编辑后：212 题截断，940 题正常结束，其中 696 题答对（74%）。
- **小心选择偏差**：原始模型基本只在简单题上能正常结束，所以两个"结束后正确率"不能直接比较。

### 2.5 运行间方差（[B] `exp8_math/eval/RESULTS_excerpt.md`）

thinking 开启的 Qwen 一共有 7 次完成的运行（MATH-500）：

- 0 次 forward（v2）：7.9，没有编辑；
- 5 次（v3）：7.9，没提交编辑；
- 10 次（v3）：12.3；
- 25 次（v3）：14.1；
- **50 次（v3）：48.1**；
- 10 次（v4）：7.9，没提交编辑；
- 25 次（v4）：14.1。

另有两个没提交的候选，事后评测分别为 8.6 和 10.1。**只有 p50c 一次超过 15%。**

### 2.6 作者代码注释里的 thinking-off 信息（[B] `code/profiles.py` 第 23–26、37、121 行）

这些只是代码注释，没在博客或结果表里公开，**不能当作结论直接引用**，只能作为动机，需要我们自己测。

- thinking 开启时 0.8B 会循环（"Wait, let's check ..."），92% 的 MATH-500 回答撞到 32k 上限，准确率 9.3%。
- 作者也用 thinking 关闭的设置跑过编辑实验，用的是 card 里的 non-thinking 基准参数：t 0.7，top_p 0.8，top_k 20，presence 1.5，最多 8192 token。
  - 未编辑模型在两次各 128 题的练习 pass 上分别得 **55.5% 和 50.8%**；
  - 注释原文："v3/v4 gains sat below the noise"。

### 2.7 Gemma 3 270M 的追加实验（[A]，[B] `exp8_math/gemma_followup/`）

Gemma 基本不循环，只有 8.5% 的回答撞上限。Astra 没能提升它：MATH-500 从 6.8 最多到 7.7。

### 2.8 Qwen3.5-0.8B 的 model card 要点（[C]，行号指 README.md）

- **架构**：
  - 24 层，hidden 1024，布局为 6 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))；
  - FFN 中间维度 3584；
  - Gated Attention：8 个 Q 头、2 个 KV 头，head dim 256；
  - DeltaNet：16 个头，head dim 128；
  - 词表 248,320，输入 embedding 与输出层共享权重。
- **默认模式**（约第 585 行）："Qwen3.5-0.8B operates in non-thinking mode by default"。
- **推荐采样参数**（约第 705–708、931–940 行）：

  | 场景 | temperature | top_p | top_k | presence_penalty |
  |---|---|---|---|---|
  | thinking，文本任务 | 1.0 | 0.95 | 20 | 1.5 |
  | thinking，图文 / 精确编程 | 0.6 | 0.95 | 20 | 0.0 |
  | non-thinking，文本任务 | 1.0 | 1.00 | 20 | 2.0 |
  | non-thinking，图文 | 0.7 | 0.80 | 20 | 1.5 |

  四种场景都是 min_p 0、repetition_penalty 1.0。
- **基准评测用的参数**：
  - 语言类 benchmark（约第 285 行）：t 1.0，top_p 0.95，top_k 20，presence 1.5；
  - 图文类 benchmark（约第 577 行）：thinking 用 t 0.6，其余同上；non-thinking 用 t 0.7，top_p 0.8，top_k 20，presence 1.5。
  - → 原文的 `think35`（t 0.6 + presence 1.5）对应的是**图文**基准的 thinking 设置，不是语言类的（t 1.0）。
- **第 856 行警告**：thinking 模式下用推荐参数时，0.8B 比其他 Qwen3.5 模型更容易陷入思考循环、停不下来；官方建议针对用途调参，并用流式生成及时检测和中断。
  - → 我们的 D3（循环熔断）就是在照官方建议做。
- **官方分数（0.8B）**：IFEval non-thinking 52.1，thinking 44.0；MMLU-Pro non-thinking 29.7，thinking 42.3。官方**没有**报 MATH-500 和 GSM8K。
- **chat template**：
  - thinking 开启时，生成提示以 `<think>\n` 结尾；
  - thinking 关闭时，以 `<think>\n\n</think>\n\n` 结尾；
  - 强制结束思考时，追加 `\n</think>\n\n`。
  - 动手前请在 `chat_template.jinja` 里再确认一次。

---

## 3. 研究问题与预注册草案

### RQ1：基线公平吗？（P0，决定论文叙事）

**模型**：base（固定 revision `2fc0636…`）、p50c。p10c、p25c 是 P2，可选。

**解码配置**：唯一定义处放在 `src/wsprev/decoding.py`，其他地方一律引用。

| 编号 | thinking | temperature | top_p | top_k | presence | 说明 |
|---|---|---|---|---|---|---|
| D0 | 开 | 0.6 | 0.95 | 20 | 1.5 | 原文的 `think35` |
| D1 | 开 | 1.0 | 0.95 | 20 | 1.5 | card 推荐的 thinking 文本参数，也是语言类 benchmark 参数 |
| D2a | 关 | 1.0 | 1.00 | 20 | 2.0 | card 推荐的 non-thinking 文本参数 |
| D2b | 关 | 0.7 | 0.80 | 20 | 1.5 | card 的 non-thinking 基准参数，等于作者的 `nothink35` |
| D3 | 开 | 同 D0 | | | | D0 + 循环熔断（见下） |
| D4（P2） | 开 | 同 D0 | | | | 思考预算固定 8192 token，到点强制结束（s1 式 budget forcing） |

- 所有配置：min_p 0，seed 0，max_tokens 32768。关思考的配置也用 32768，让各配置的截断口径一致；"在 8192 处截断"的情况可以从数据里直接算出来。
- **D3 的规格**（规格写进 prereg，阈值只能在练习集上标定）：
  1. 思考作为**一个不中断的请求**生成（与 D0 是同一个请求），熔断器在思考每达到 1024 的整数倍 token 时检查一次；
     - （2026-11-03 代理提议，**用户已确认并写入注册的 prereg**）原写法是"每次生成 1024 token，开 prefix caching 续写"。但 vLLM 的 presence penalty 只统计本请求自己生成的 token，分段续写会让惩罚每 1024 token 清零一次，D3 的思考就不再和 D0 相同。实现见 `src/wsprev/generate.py`（vLLM 的 step 循环），细节见 prereg 第 3 节。
  2. 每个检查点，在思考文本的最近 2048 个 token 里计算 dup8，即"在该窗口内此前已出现过的 8-gram"所占比例；
  3. 满足以下任一条件就熔断：dup8 ≥ τ，或者思考长度达到 30,720；
  4. 熔断后追加 `\n</think>\n\n`，再用同样的采样参数生成最多 2048 token 作为答案；presence penalty 用 logit bias 延续到答案阶段（同上，已确认）；
  5. 每条记录熔断位置、原因和是否触发。
  - τ 只在练习集的 DEV 子集上标定：取熔断后练习准确率最高的值，候选为 {0.3, 0.5, 0.7}。参考：Astra 统计的原始模型截断样本尾部 dup8 均值是 0.958。
- **评测集与样本数**：
  - MATH-500 每题 2 个样本，GSM8K（1319 题）每题 1 个，IFEval（541 题）每题 2 个。
  - D0 只额外跑 MATH-500 每题 1 个样本，用于校准评测流程，同时给生存分析提供数据。
  - base 和 p50c 在 D0 下的完整结果已经在 [B] 里。
- **校准闸门**：在 D0、MATH-500、每题 1 个样本下：
  - base 应在 [4.5%, 11.5%] 之间，截断率约 93%；
  - p50c 应在 [43%, 55%] 之间，截断率约 33%。
  - 落在区间外就**停下来查评测流程**：vLLM 版本、chat template、enable_thinking、提示字符串、math-verify 版本、用纯文本版还是多模态 checkpoint。查清楚之前不跑 RQ1 全量。

**预注册的比较**（主指标：MATH-500 准确率，按题配对；次要指标：GSM8K、IFEval）。**已注册（2026-11-03），以 `prereg/criteria.md` 为准**；下面是初稿，与注册版不同处已注明。

- 编辑效应 Δ_c = Acc(p50c, c) − Acc(base, c)，c ∈ {D0, D1, D2b, D3}。
  - H1（复现）：Δ_D0 ≥ 30pp。
  - H2：Δ_D1 < Δ_D0，即换成官方参数后增益变小。
  - H3：Δ_D3 < 5pp，即循环熔断能替代编辑。初稿写的是"或其 95% CI 包含 0"；注册版改为非劣效检验：单侧 bootstrap p = P*(Δ*_D3 ≥ 5pp)，Holm 校正后 < 0.05 才算成立。
  - H4：Δ_D2b < 5pp，即在默认的 non-thinking 模式下编辑没有增益；判据同 H3。
- 交叉比较：Acc(p50c, D0) − Acc(base, c*)，c* 是 base 在 {D1, D2b, D3} 中最好的配置。
  - c* 在练习集的 SELECT 子集（128 题，`results/splits.json`）上选定，**不能在 benchmark 上选**，否则有选择偏差。
- 统计方法：
  - 对题目做配对 bootstrap，10,000 次，报 95% CI；
  - 对 H2–H4 三个检验做 Holm 校正；
  - 每个配置都同时报准确率、截断率、P(答对 | 正常结束)、平均 token 数。
- 结论模板（三种结果都要能写）：
  - H3 和 H4 成立 → "增益主要源于终止失败，可被解码层面的手段替代"；
  - 两者都不成立 → "编辑在公平基线下仍有实质增益"；
  - 其他情况 → 如实报告各效应量。

### RQ2：不用 LLM 能找到同样的修复吗？（P1）

全程不调用任何 LLM。**benchmark 绝不参与任何选择。**

> **以 `prereg/rq2_rules.md` 为准**（2026-11-05 写成，任何 RQ2 生成之前）。它把下面的初稿定成了可执行的规则，主要的细化：方向用 DEV 前 `dev_problems` 题，筛选用 DEV[448:512]（不参与求方向）；正常窗口要求 dup8 ≤ 0.3；AUC 交叉拟合；排序规则和并列时的次序；筛选时也跑 base、p50c 作参照（不参选）；精简 / 完整两个方案（表见规则文件第 8 节）。代码：`src/wsprev/rq2.py`、`directions.py`。

1. **划分练习集**：固定种子，从练习题里切出 DEV（512 题）、HOLDOUT（256 题），以及 RQ1 用来选配置的 128 题，三者互不重叠。划分写入 `results/splits.json`。
2. **准备循环与正常的上下文**：
   - 让 base 按 D0 在 DEV 上生成；
   - "循环窗口"：截断回答的尾部 2048 token，且 dup8 ≥ 0.8；
   - "正常窗口"：同一条回答开头的思考段，这样可以控制题目内容。另外加一组"正常结束回答的中段"作为对照。
3. **teacher-forced 读激活**：用 HF transformers 加 forward hook，在 GPU 上跑。
   - 取第 l 层输出处的残差均值差，归一化得 u_l；扫描 l ∈ [8, 22]，按投影的可分性（AUC）选出前 2 层；同时保留 Astra 用的 {15, 19}，作为一组单独的层设置。
   - 取 down_proj 的输入激活 a ∈ ℝ³⁵⁸⁴。
4. **给神经元打分**：c_j = (uᵀw_j)·(E_loop[a_j] − E_prod[a_j])，其中 w_j 是 W 的第 j 列。按 c_j 取前 k 列。理由见 RQ3 的命题 3。
5. **编辑**：W' = W − α·u (m_k ⊙ (uᵀW))。网格：层组合 ∈ {Astra 的 {15,19}，AUC 前 2 层}，k ∈ {32, 64, 128, 3584}，α ∈ {1, 2, 4, 6, 8}，共 40 组。
6. **筛选**：
   - 每组配置在 64 道 DEV 题上用 D0 生成，上限 8192 token；指标 = 8192 内正常结束的比例与准确率。
   - 取前 5 组，在 HOLDOUT（256 题，上限 32k）上评测，按准确率选出最终编辑。
   - forward 次数约 40 + 5 + 少量 teacher-forced 读取，不超过 Astra 的 50 次预算。
7. **最终评测**：在 RQ1 的 D0 和 D3 下评测自动编辑。
8. **对照**（MATH-500，每题 1 个样本，D0）：
   - (a) 同样的列、同样的 α，换成随机单位方向；
   - (b) 同样的 u，随机选 64 列；
   - (c) Astra 选的列，但 α=1（纯投影）。
9. **成本对比**：我们报 GPU 小时数和 0 个 LLM token。Astra p50c 用了 158 轮、80 次 Python 运行、4 段对话，墙钟约 20.3 小时，其中 4.7 小时是在等服务恢复。

### RQ3：理论与生存分析（P0/P1，用户主导，代理协助）

设 MLP 输出 y = W a，其中 a ∈ ℝ³⁵⁸⁴，W ∈ ℝ^{1024×3584}；u 是单位向量，m ∈ {0,1}³⁵⁸⁴，S = {j : m_j = 1}。

- **命题 1（等价性）**：W' = W − α·u (m ⊙ (uᵀW)) 等价于 W'a = Wa − α·u·uᵀW(m ⊙ a)。
  - 由此：y 中与 u 正交的分量不变；uᵀy' = uᵀy − α·uᵀW_S a_S。
  - 当 m = 1 时，uᵀy' = (1−α)·uᵀy：α=1 是投影（Arditi et al. 2024 的方向消融），α=2 是反射，α=6 是取反后放大 5 倍。
  - 推论：权重编辑等价于在推理时，对该层 MLP 输出加一个随激活变化的 steering 项。**写单元测试验证**：权重版和 hook 版在随机输入上输出一致（bf16 误差范围内）。
- **命题 2（扰动上界）**：‖Δy‖ = α·|uᵀW_S a_S| ≤ α·‖W_Sᵀu‖·‖a_S‖。
  - 再实测下一层残差的变化和下一 token 分布的 KL，分别在循环上下文和正常上下文上测。
  - 参考：Astra 报告编辑在 α = 1/2/4 时，重复分支的 log p 分别下降 0.024、0.049、0.104 nats/token，正常分支几乎不变。
- **命题 3（选列准则）**：一阶近似下，编辑让"循环与正常上下文在 u 方向上的投影差"减少 α·Σ_{j∈S} c_j。所以在 |S| = k 的约束下，按 c_j 取前 k 个能让这个减少量最大（线性目标的贪心最优）。
- **生存分析**：
  - 设 T 为生成到 `</think>` 为止的 token 数（non-thinking 模式用 EOS），在 L = 32768 处右删失。
  - 对每个（模型、配置、benchmark）组合，用 Kaplan–Meier 估计 Ŝ(t)，用 Greenwood 公式给 CI，用 Nelson–Aalen 平滑得到风险函数 ĥ(t)。
  - **准确率分解**：Acc = P(T ≤ L)·P(答对 | T ≤ L) + P(T > L)·P(答对 | T > L)。用两因素的 Shapley 分解，把 Δacc 拆成"终止效应"和"条件正确率效应"。
  - **选择偏差**：另做一组配对分析，只看两个模型都能正常结束的题。
  - **预算曲线**：Acc(L) 在 L ∈ {4k, 8k, 16k, 32k} 下的值，可以从现有数据直接算出。
- P2：用"正常 ↔ 循环"两状态马尔可夫模型拟合终止风险，解释 10/25/50 次 forward 结果的阶跃式差异。

### RQ4：可选（P2），默认不做，需用户批准 API 预算

用作者公开的实验 1 网络（[B] `exp1_which_game/networks/*.npz`、`*_actor.txt`、`description.txt`），问 Claude、GPT、DeepSeek "这个网络是玩什么游戏的"。

- 对比原网络和两种不改变函数的变换：
  - 打乱隐藏单元的顺序；
  - 对 tanh 单元做符号翻转：入权重、偏置、出权重同时取反。
- 再加一个同形状的随机初始化网络作对照。
- 提问格式完全照抄 `exp1_which_game/runs/*/turns/t001/prompt.txt`。

---

## 4. 实验矩阵与算力

token 数由 `scripts/estimate_tokens.py` 估算（2026-11-04），结果在 `results/cost_estimate.json`。方法：原作者公开了每个模型 MATH-500 200 题、GSM8K 200 题、IFEval 100 题的第一个 D0 样本全文；用模型的分词器重新切分得到 D0 的长度（和原文公布的平均长度相差 0.1%–5%，IFEval 的 p50c 差 9%；这些文本只是部分题目的第一个样本）；再在这些思考文本上回放注册的 D3 熔断器，得到 D3 的长度。只读长度和熔断位置，不读对错。D1 假定和 D0 一样长；D2a / D2b（不思考）没有数据，假定每条 0.8k–2.5k token；熔断后的答案按上限 2,048 token 计。

| 优先级 | 阶段（`scripts/run_stage.py`） | 约 token 数 |
|---|---|---|
| P0 | smoke + calib：D0，MATH-500 500 题 × 1 样本，两个模型 | 26.5M |
| P0 | tau：base，练习集 DEV 前 256 题 × 3 个 $\tau$ | 6.7M |
| P0 | select：base，练习集 SELECT 128 题 × D1 / D2b / D3 | 5.4M |
| P0 | rq1，MATH-500：D1 53.1M、D2a 5.0M、D2b 5.0M、D3 30.3M | 93.4M |
| P0 | rq1，GSM8K：D1 55.8M、D2a 2.1M、D2b 2.1M、D3 26.2M | 86.2M |
| P0 | rq1，IFEval：D1 31.5M、D2a 1.7M、D2b 1.7M、D3 11.2M | 46.1M |
| P0 | 合计：注册的完整设计 | 264M |
| P1 | RQ2：原计划规模；精简版约 30M | 135M |
| P2 | D4、p10c / p25c、RQ4 | — |

- 钱主要花在 base 开着思考、一路循环到 32k 上：D0 / D1 每个样本约 3 万 token；D3 只要约三分之一（base 在 MATH-500 上，取 $\tau = 0.7$ 时约 1.1 万）。
- **吞吐量**（每张卡的 token/s）：解码受显存带宽限制，按第三方复现在 H200 上的实测值（约 1.42 万）按带宽折算，误差 ±50%，冒烟测试后用实测值重算：3090 / 4090 约 2.5k（1.5k–3.5k），5090 约 5k，H100 SXM 约 10k，H200 约 14k。
- **预算上限 100 元**（用户，2026-11-04）。H 系列跑得快，但每个 token 贵十倍以上：$7.27/时的 H200 约 ¥1.0 / 百万 token，¥0.80/时的 3090 约 ¥0.09 / 百万 token。所以用同一台机器上的 2–4 张 3090（或 4090），`run_stage.py` 按卡数自动分片。按 ¥0.8–1.0 / 卡时：P0 约 ¥25–30（吞吐量最差时约 ¥50），加完整 RQ2 约 ¥40（最差约 ¥75），都含约 3 卡时的装环境开销。
  - **2026-11-04 更新**：实际租的是恒源云 2 × 3090，¥3.0/时（¥1.5 / 卡时）。用户把上限提到 **150 元**，不加预算规则 Amendment（按注册的完整设计跑）。按上面的吞吐量估计：P0 约 15 小时、¥44（最差 24 小时、¥73），加完整 RQ2 约 ¥70（最差约 ¥115）。冒烟测试后用实测 token/s 重算。
- **必要时的省钱顺序**：先把 RQ2 缩成精简版（不涉及预注册）；再砍 GSM8K / IFEval 上的 D1（−87M，要先改预注册）。MATH-500 的样本数不砍：H3 / H4 要证明效应小于 5 个百分点，样本本来就不富裕。
- **评估过但不采用的提速办法**：FP8 KV cache（改变数值，和原评测不再可比）；用 Qwen3.5 自带的 MTP 头做 speculative decoding（理论上分布不变，但和 presence penalty、logit bias 一起时 vLLM 是否严格等价，事先无法验证）；三个 $\tau$ 共用一条思考流（只省约 3.5M token）。
- **用卡**（2026-11-04）：AutoDL 无卡。比价站 gpuprice.dev（2026-11-03 23:15）：3090 晨涧云 ¥0.80、恒源云 ¥0.99、智星云 ¥1.10；4090 算家云 ¥1.24、晨涧云 ¥1.38，均标"充足"，以平台页面为准。有"无卡模式"的平台先用它装环境、下模型。
- **vLLM 设置**：bf16；`max_model_len = 32768 + 2048`；`gpu_memory_utilization = 0.9`；seed 0。版本先对齐第三方复现（vLLM 0.33.0、transformers 5.20.2、math-verify 0.9.0），对不上时记录实际版本。
  - 硬条件（2026-11-04 在 PyPI 核实）：vLLM 0.33.0 要求 Python ≥ 3.11，固定 torch 2.14.0；这一版 torch 是 CUDA 13 构建，所以驱动要支持 CUDA 13.0（R580+），GPU 至少 sm_75，bf16 要 sm_80+。V100 / P100 / T4 不能用。vLLM 0.33.0 没有 CUDA 12 构建（0.32.0 还有 `+cu129`）；万一租不到 R580 驱动的机器，退路是 vLLM 0.32.0+cu129，但这偏离了注册的版本，要先写 Amendment。
  - prefix caching：D3 改为不中断生成后不再需要；默认沿用作者 `evaluate.py` 的 vLLM 参数（不设），需要时用 `--vllm-arg enable_prefix_caching=true` 打开。
- **分片**：按题目编号 i::k 切分，每片单独输出，支持断点续跑。实例中途掉线不能丢已完成的结果。
- **国内网络**：模型从 ModelScope 或 hf-mirror 下载；AutoDL 上用 `source /etc/network_turbo` 打开学术加速。下载后对照 Hugging Face 上的哈希做校验。机器访问不了 GitHub 时（作者日志仓库、NLTK 的 punkt_tab 都在 GitHub 上），由代理从用户电脑用 scp 拷过去。

---

## 5. 仓库结构与代码约定

```
wsp-revisited/
  CLAUDE.md                 指向本文件
  README.md
  docs/plan-for-agents.md   本文件
  docs/guide-for-me.md      给用户的操作指南（别改它的结构，只更新状态）
  prereg/criteria.md        RQ1 预注册，正式运行前提交
  third_party/wsp_logs/     从 [B] 拷贝的 MIT 脚本（mathqa.py, apply_edit.py, st_numpy.py），未修改，附 LICENSE 和出处
  third_party/ifeval/       google-research 的 IFEval 官方检查器（Apache-2.0），改动见其 README
  src/wsprev/
    decoding.py             D0–D4，唯一定义处
    generate.py             vLLM 生成（每个样本一个请求，step 循环里做循环熔断），分片和断点续跑
    loopdetect.py           dup8 等重复度指标
    grade.py                调用 mathqa（math-verify）判分；IFEval 用官方检查器
    data.py                 读 benchmark 和练习集划分
    edits.py                列掩码秩 1 编辑、随机对照、hook 版等价实现
    _vendor.py              载入 third_party 代码
    _vllm_ext.py            vLLM worker 扩展：把 logit_bias 每请求 1,024 个的上限提到 32,768（D3 的惩罚延续需要；prereg Amendment 2）
    summary.py              读输出目录、汇总指标、校准闸门（summarize.py / run_stage.py / rq1.py 共用）
    rq1.py                  RQ1 的注册分析：配对 bootstrap、Holm、结论措辞；τ 和交叉比较配置的选择规则
    directions.py           RQ2：窗口、读激活（HF hook）、求方向、交叉拟合 AUC、给神经元打分、编辑网格、从 p50c 反推 u 和列
    rq2.py                  RQ2 的各步：plan / read / build / swapcheck / screen / choose / holdout / export / report / estimate
    _vllm_swap.py           vLLM worker 扩展：原地替换、恢复 down_proj 权重（RQ2 筛选时一个引擎跑完所有编辑）
    survival.py             RQ3：Kaplan–Meier + Greenwood、Nelson–Aalen 与平滑风险、Shapley 分解、配对单元格、预算曲线；读本地 runs/ 的记录 → results/rq3/、results/rq3_summary.md
  scripts/
    env.sh                  GPU 机器上的路径和环境变量（自动找数据盘：AutoDL、恒源云 /hy-tmp、/workspace，否则 ~/wsprev）
    setup.sh                一次性准备（平台无关）：检查、环境、作者日志仓库、模型下载与哈希校验、重建 p50c、benchmark
    check_vllm.py           GPU 自检：logit_bias、abort、种子约定、D3 熔断端到端
    run_stage.py            按阶段跑：smoke、calib、tau、select、rq1；自动按 GPU 数分片，检查前置条件
    check_ifeval_grader.py  用我们的 IFEval 判分器重判作者公开的 IFEval 文本，和原评测逐题对照 → results/ifeval_grader_check.json
    run_all.sh              无人值守按顺序跑完 自检 → smoke → calib → 预算检查 → tau → select → rq1，出错即停，每阶段在本机提交小结果
    choose.py               在练习题上选 τ（可写回 decoding.py）和交叉比较的 base 配置
    build_p50c.py           原地重建 p50c（低内存，无卡模式可跑），逐张量 sha256 校验
    verify_model.py         对照 Hugging Face 的哈希校验模型文件，写 wsprev_model.json
    prepare_data.py         下载 MATH-500 / GSM8K / IFEval，记录数据集 revision 和哈希
    make_splits.py          练习集划分 → results/splits.json（已生成并提交）
    summarize.py            汇总准确率、截断率等，对照原文同题结果，检查校准闸门
    plot_rq3.py             RQ3 的图：只读 results/rq3/ 下的小文件 → results/rq3/figs/（需要 matplotlib）
  results/                  只放小型汇总（summary.json、每题一行的 CSV）；大文件不进库
  paper/                    LaTeX 源文件
  tests/
```

**每条生成记录一行 JSONL**，字段：

- `benchmark, problem_id, model, model_sha, config, sample_idx, seed`
- `n_gen_tokens, think_closed, think_tokens, finish_reason`
- `loop_triggered, trigger_pos, dup8_tail, pred, correct`
- `text`：gzip 后另存，不进 git

**每次运行写 `results/<run_id>/meta.json`**：git commit、各库版本、GPU 型号、起止时间、总 token 数、token/s、完整命令行。

**必备测试**：

- 命题 1：权重版与 hook 版输出一致；
- 循环检测器在合成字符串上的表现；
- 判分器在已知答案上的表现；
- `apply_edit.py` 的 sha256 校验通过；
- 同一 seed、同一配置重复跑一次，结果一致。

---

## 6. 里程碑与闸门

T0 = 用户宣布开工的时刻。

| 里程碑 | 时间 | 内容 | 闸门 |
|---|---|---|---|
| M0 | T0+3h | 冒烟测试所需代码、`scripts/setup.sh`、`prereg/criteria.md` 推送完毕 | 用户审阅并确认 prereg |
| M1 | T0+5h | 冒烟测试与 D0 校准 | 落在 §3 的校准区间内，否则停下排查 |
| M2 | T0+16h | RQ1 的 P0 部分完成，写出 `results/rq1_summary.md` | 和用户一起对照 prereg 定下叙事 |
| M3 | T0+30h | RQ2 完成 | — |
| M4 | T0+34h | RQ3 的分析与图 | — |
| M5 | T0+44h | 全文初稿 | 用户逐段改写、核对所有数字 |
| M6 | T0+48h | 发布：先邮件原作者，再公开 GitHub、Zenodo；arXiv 等背书 | 用户最终确认 |

**时间不够时的砍法**（按顺序）：RQ4 → D4 → p10c/p25c → 缩小 RQ2 网格 → GSM8K 改为 500 题子集。

---

## 7. 论文大纲与图表

1. **引言**：WSP 的主张；我们的问题；主要发现。最后再写。
2. **背景**：实验 8 的设置；方向消融与 steering。
3. **公平基线（RQ1）**：表 1 是主结果；图 1 是 base 与编辑后模型在 D0 和 D1 下的生存曲线。
4. **不用 LLM 的基线（RQ2）**：图 2 是网格热力图；表 2 是与 Astra 的效果和成本对比。
5. **理论与分解（RQ3）**：命题 1–3；图 3 是准确率分解柱状图；图 4 是准确率随思考预算变化的曲线。
6. **讨论与局限**：
   - 只研究了一个模型；
   - 作者的 thinking-off 结果只见于代码注释；
   - 我们没有调用 Astra。
7. **附录**：
   - 全部配置与结果；
   - prereg 原文；
   - 表 3：作者 7 次运行的方差（来自 [B]）；
   - 复现说明。

**待核实的相关工作**。以下 arXiv ID 来自记忆，引用前逐一打开核对：

- Arditi et al. 2024，refusal 由单一方向介导，2406.11717
- Muennighoff et al. 2025，s1 / budget forcing，2501.19393
- Turner et al. 2023，ActAdd，2308.10248
- Zou et al. 2023，Representation Engineering，2310.01405
- Meng et al. 2022，ROME，2202.05262
- Holtzman et al. 2019，神经文本退化，1904.09751
- Welleck et al. 2019，unlikelihood training，1908.04319
- Elhage et al. 2021，A Mathematical Framework for Transformer Circuits，transformer-circuits.pub
- Kaplan & Meier 1958

另外还要专门检索 2025–2026 年关于"小推理模型的循环与重复"的工作。

---

## 8. 风险与对策

- **复现不出 D0 的数字** → 先逐项对齐第三方复现 [D] 的全部设置，再找差异。
- **新架构的 kernel 不支持当前 GPU 或 vLLM 版本**（Gated DeltaNet）→ 用 vLLM 0.33.0；仍不行就换卡型，Hopper 系列优先。
- **D3 被质疑是"调出来的"** → τ 只在练习集上标定，并写进 prereg；报告触发率和触发位置的分布。
- **配置选择偏差** → 只用练习集 HOLDOUT 选配置，benchmark 只评测一次。
- **结论说过头** → 每条结论都对照 prereg 的措辞模板；局限性单独成节。
- **实例掉线或被释放** → 结果写到数据盘，分片输出，支持断点续跑，每个分片结束就同步一份小汇总回仓库。

---

## 9. 待用户决定的事

- T0（开工时刻）；
- 最终用哪张卡；
- prereg 里的阈值（30pp、5pp）是否采用；
- RQ4 做不做（默认不做）；
- arXiv 背书的进展；
- 公开发布的时间点。

---

## 10. 与用户协作的方式

- 开工时先读 `docs/guide-for-me.md`，了解用户那边的步骤，保持两份文档一致。
- 每个里程碑结束时，用中文给用户写简短汇报：做了什么、结果数字（附文件路径）、下一步、需要用户做什么。
- 需要用户在 GPU 机器上执行的命令，给出完整可复制的命令块。长任务必须放在 tmux 里跑，并告诉用户怎么查看进度。
- 用户是数学系学生、LLM 新手：解释术语时简洁，不居高临下；数学部分让用户主导，你负责检查和补充细节。

---

## 11. 进度表（完成后打勾，并注明日期和提交哈希）

- [x] M0 仓库骨架、`decoding.py`、`generate.py`（含 D3）、`grade.py`、测试（2026-11-03，`4d3fe37`；本地 CPU 测试全部通过，vLLM 部分还没在 GPU 上跑过）
- [x] M0 练习集划分 `results/splits.json`（2026-11-03，`4d3fe37`，种子 20261103；DEV 512 / HOLDOUT 256 / SELECT 128）
- [x] M0 跑阶段的脚本、GPU 自检、τ / 配置选择、RQ1 注册分析代码（2026-11-03，与注册同一个 PR）
- [x] M0 `scripts/setup.sh`（平台无关：装环境、下模型、校验哈希、重建 p50c）：2026-11-04 在恒源云 2 × 3090 上首次跑通（vLLM 0.33.0、transformers 5.20.2、torch 2.14.0、math-verify 0.9.0；p50c 改动 65,260 + 64,928 = 130,188 个元素，哈希一致；80 个单元测试通过）。机器上 GitHub 克隆失败，作者日志和 NLTK punkt_tab 由本地 scp 过去
- [x] M0 `prereg/criteria.md` 用户确认（2026-11-03，5 个决定全部按草案建议）→ 注册提交：见下一行
- [x] M0 注册提交：`d06c4750112c2526ebb006f78ef94c48d6e525dc`（2026-11-03，作者 cyj-hashtag；之后 prereg 只能追加带日期的 Amendment）
- [x] M0 prereg Amendment 1（2026-11-03，用户审阅后要求；任何生成之前）：公式改为 LaTeX、加符号表、澄清 D3 的 token 上限和若干措辞；补上第 5 节登记了但 `rq1.py` 没算的 GSM8K / IFEval 效应量（`5b5edf2`）。公式已用 GitHub 渲染接口和 MathJax 3.2.2 逐个核对
- [x] 仓库公开（2026-11-04，用户要求）：旧私有仓库改名为 `cyj-hashtag/wsp-revisited-archive`（保持私有），新建同名公开仓库，只推送 `main`，提交哈希不变（注册提交仍是 `d06c475`）。原因：GitHub 仍保存旧仓库里改写前的提交，直接公开会暴露。GPU 机器不放 GitHub 凭据，结果拷回本地提交
- [x] M1 GPU 自检 8 项全部 PASS（2026-11-04，`923a8f3`，`results/preflight.json`）
- [x] M1 冒烟测试（2026-11-04，`d99a6ed`，`results/smoke/`）：base 7.0%（截断 94.0%）；p50c 57.0%（SE 5.0，截断 25.0%），比区间上限 55% 高 2pp，流程因此停下。同 100 题上原评测为 53.2%，逐题配对差 +3.8pp（SE 3.6），按原评测的对题数分层也无系统偏差 → 判为小样本波动。注册的闸门在 500 题的 calib 上（`run_stage.py` 也只在 calib 上强制），所以 `run_all.sh` 改为 smoke 的闸门只作参考（`0bb1d7f`）。实测 token/s：每卡 2,468（base）/ 2,586（p50c）
- [x] M1 D0 校准通过（2026-11-04，`7633431`，`results/calib/`）：base 8.2%（SE 1.2，截断 92.6%；原评测 7.9%），p50c 51.0%（SE 2.2，截断 29.8%；原评测 48.1%，第三方复现 51.0%）。两卡合计 4,896 token/s；预算检查：已花约 ¥8，剩余 P0 约 15.2 小时 ≈ ¥46
- [x] tau / select（2026-11-04，`f7493e4` / `3da1ef8`）：三个 τ 在练习 DEV 前 256 题上都答对 138 题 → 按规则取较大的 τ = 0.7（逐题对错并不相同，平手是巧合）；SELECT 上 D3 53.9%、D2b 52.3%、D1 30.5% → 交叉比较用 D3
- [x] RQ1 中断与修复（2026-11-04）：11:14 p50c 的 D3（MATH-500）一个答案请求带 1,054 个 logit bias，超过 vLLM 0.33.0 的 1,024 上限，引擎退出。`a43c1aa` 用 `_vllm_ext.py` 把上限提到 32,768，自检加第 9 项；重跑自检 9/9 通过，第 1–8 项输出与修复前一致。12:25 断点续跑，之前的结果都不受影响。用户要求记为 prereg **Amendment 2**（同日，见 prereg 末尾，也记了 smoke 闸门一事和 Amendment 1 以来的全部代码改动）
- [x] IFEval 判分器对照（2026-11-04，`results/ifeval_grader_check.json`，`python scripts/check_ifeval_grader.py`）：重判作者公开的每模型 100 道 IFEval 第一样本，base 100/100、p50c 99/100 与原评测一致；唯一不同的 `ifeval_1389`（全大写 + 字母 o ≥ 40 次）是官方检查器先转小写再计数。起因：base 在 D1 下 IFEval 58.8%，高于 card 的官方分数（thinking 44.0、non-thinking 52.1）；card 的 IFEval 协议 TODO(verify)
- [x] M2 RQ1 P0 完成 → `results/rq1_summary.md`（2026-11-04 19:26 生成完毕，`27ab278`；注册分析 `820eaf4`，`python -m wsprev.rq1`）。MATH-500：Δ_D0 42.8、Δ_D1 8.1、Δ_D2b 2.3（95% CI [−1.0, 5.6]）、Δ_D3 16.7 pp；H1、H2 成立，H3、H4 不成立（H4：p = 0.0555，Holm 后 0.111）→ 注册的结论模板："The edit retains a substantial gain under the fair baselines we tested." 交叉比较 Acc(p50c, D0) − Acc(base, D3) = 12.8 pp [9.3, 16.3]。原始记录和全文在本地 `runs/gpushare_2026-11-04/`（不进 git，sha256 见同目录）。`results/smoke/` 已恢复为 100 题版本（`9219118`）
- [ ] M2 与用户确定叙事分支
- [x] M3 RQ2 代码与 CPU 测试（2026-11-05）：`src/wsprev/directions.py`、`rq2.py`、`_vllm_swap.py`，`scripts/run_stage.py` 加了 rq2-dev / rq2-read / rq2-screen / rq2-holdout / rq2-final 五个 stage；规则文件 `prereg/rq2_rules.md`；38 个新测试（含在一个按 0.8B 层布局的微型随机 Qwen3.5 上验证读激活的 hook，与 `hidden_states[l+1]` 逐元素一致），全部 118 个测试通过。费用估算 `python -m wsprev.rq2 estimate` → `results/rq2/cost_estimate.json`：精简版 34–45M token、1.9–4.3 小时、¥6–13；完整版 114–155M token、5.8–14 小时、¥17–42（数据盘被清空要重跑 setup 时各加约 ¥3）。GPU 上还没跑过：vLLM 原地换权重靠 swapcheck 把关，HF 读 33k token 上下文的显存靠 `--max-context` 兜底
- [x] M3 RQ2 规则与方案（2026-11-05，任何 RQ2 生成之前）：规则文件 `prereg/rq2_rules.md` 在 `9964c0c`；用户选**完整版**，`results/rq2/plan.json` 在 `2511b27`；用户同意后推送到 `origin/claude/gpushare`（`2511b27`，未合并进 `main`）。无人值守脚本 `scripts/run_rq2.sh`（`7cc5de4`，未推送）
- [ ] M3 RQ2 在 GPU 上跑完
- [x] M4 RQ3 生存分析与分解（2026-11-05，代码 `b3c324b` / `6f771db`，结果 `48a8ae3`；`python -m wsprev.survival`，再 `python scripts/plot_rq3.py`；`results/rq3_summary.md`、`results/rq3/`、图在 `results/rq3/figs/`；17 个测试，两次运行输出逐字节相同）。MATH-500 上 Δ 的两因素 Shapley 分解（终止 / 条件正确率，pp）：D0 +53.9 / −11.1，D1 +12.6 / −4.5，D3 +23.3 / −6.6。按"哪个模型正常结束"配对分格：两者都结束的配对贡献 D0 +0.0、D1 −0.7、D3 +0.3，其余几乎全部来自只有 p50c 结束的配对；Shapley 的负的条件项是选择偏差（base 只在简单题上结束）。D3 下 base 熔断 931 次，其中 600 次答案又在 2,048 token 处被截断。另：同一 seed 下 D3 与 D0 的样本并不逐条相同（base 32 条未熔断样本中 0 条长度一致），vLLM 大批量下只保证同分布
- [ ] M4 待办：用户审阅 RQ3 的定义（事件、删失、"结束"的口径、Acc(L) 的下界性质，见 `survival.py` 文档字符串）；命题 1–3 的推导（用户）；命题 2 的实测（下一层残差变化与 KL，需要 GPU，可并入 RQ2 那次开机）
- [ ] M5 初稿完成（`paper/`）
- [ ] M5 用户改写与数字核对完成
- [ ] M6 邮件原作者 / GitHub 公开 / Zenodo / arXiv

### 交接：2026-11-05 之后的下一步

写于 RQ1 完成当晚，供新会话接手。用户明天全天都在：用户希望代理在后台写代码、测试 RQ2 和 RQ3，自己同时学习原理。

1. **讲解会话**（`llm-tutor`，只读，陪用户学原理）：开它时要求过"RQ1 没做完不要对 H1–H4 下结论"。现在注册分析已经完成，经用户同意后告诉它：可以带用户逐行读 `results/rq1_summary.md`。用户的自测问题有三个：base 在 D0 下为什么只有 8%；D2b 下 $\Delta = 2.3$ pp 为什么不能说"小于 5 pp"；D2a 下编辑在三个 benchmark 上都让模型变差，这可能说明什么。
2. **RQ3（不需要 GPU，先做）**：写 `src/wsprev/survival.py` 和测试：Kaplan–Meier + Greenwood、Nelson–Aalen、两因素 Shapley 分解、预算曲线 $\mathrm{Acc}(L)$。定义见第 3 节 RQ3。数据在本地 `runs/gpushare_2026-11-04/runs_records.tgz`（不进 git，sha256 见同目录；解到 `runs/` 下即可，`.gitignore` 已排除）。记录字段见第 5 节。D0 的逐样本长度只有 MATH-500 有（我们自己的 calib，每题 1 个样本）；GSM8K、IFEval 的 D0 只有原评测的逐题对错。小结果写进 `results/rq3/`，图另放。数学部分由用户主导。
3. **RQ2（需要 GPU）**：按第 3 节写 `src/wsprev/directions.py` 和对应的 stage。需要先让 base 按 D0 在练习 DEV 的 512 题上生成（约 16M token）。RQ2 没有预注册（prereg 只覆盖 RQ1），建议在跑之前把 RQ2 的选择规则写成文件并推送，以示事先确定。开机前先估算：精简版约 2–3 小时、¥6–10，完整版约 8 小时、¥25；问用户选哪个。
4. **机器**：恒源云实例（上次是 `ssh -p 24389 root@i-2.gpushare.com`，重开后地址可能变，问用户要新地址；密钥 `~/.ssh/id_ed25519_wsprev`）。2026-11-04 20:10 CST 之后关机；`/hy-tmp` 在关机 24 小时后清空，清空后要重跑 `scripts/setup.sh`：先用 scp 把作者日志的 git bundle 和 NLTK punkt_tab 传过去，因为机器上 `git clone` GitHub 会失败，也访问不了 raw.githubusercontent.com。代码写好再开机，开机即按 ¥3/时计费。到 2026-11-04 共花约 ¥56，上限 ¥150。控制台的监控图不可靠，以 `nvidia-smi` 为准。
5. **Git**：本地分支 `claude/gpushare` 上还有没推送的提交（用 `git log origin/claude/gpushare..claude/gpushare` 查看），也还没合并进 `main`。推送和合并都要用户同意。推送时用 `git -c credential.helper= -c "credential.helper=!gh auth git-credential" push`，确保用的是 cyj-hashtag 的凭据。
6. **待核实**：model card 的 IFEval 评测协议（base 在 D1 下 IFEval 58.8%，高于 card 的 thinking 44.0；D2b 52.7% 与 card 的 non-thinking 52.1 接近）；第 7 节列出的全部引用。
