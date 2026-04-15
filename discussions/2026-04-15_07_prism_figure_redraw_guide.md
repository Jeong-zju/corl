# PRISM 改图说明（通俗版）

这份说明不是在重复公式，而是想回答一个更实际的问题：

**你现在这张图，读者会怎么看？**

以及

**如果想让读者一眼看懂 PRISM 真正在做什么，这张图应该怎么改？**

你可以把这份文档当成“重画说明书”。

---

## 1. 先用一句大白话说清楚 PRISM 是什么

PRISM 不是一个单独做动作决策的新 policy。

它更像是一个**历史条件模块**：

- 一边看机器人到目前为止走过的**状态轨迹**
- 一边看当前时刻的**图像和状态**
- 把这些信息写进一个**固定大小的记忆槽**
- 再把这个“历史条件”喂给原本的 action expert

所以，PRISM 的角色不是“替代 ACT / diffusion / autoregressive policy”，而是：

**给这些 expert 提供一个更好的历史条件。**

如果把这件事画错，读者就会以为：

- PRISM 自己就是 policy 主体
- action expert 只是末端小头
- 当前观测不重要，主要靠 memory 出动作

这不是你想表达的意思。

---

## 2. 你改图时，心里先固定一个正确故事

你可以强行把整张图理解成下面这句话：

> 当前观测负责“现在看到什么”，状态前缀 signature 负责“之前走过什么路”，slot memory 负责“把前面的重要历史压缩记住”，action expert 负责“根据当前情况和历史条件产生命令”。

如果一张图能让读者自然读出这句话，它就基本对了。

如果读者看完后更像在想下面这些：

- 这是不是在存历史图像帧？
- 这个棱锥是不是在同时处理视觉和状态？
- action expert 是不是只是接 memory，不接当前 observation？
- 右边这些 policy 是不是并列运行？

那就说明图在表意上已经偏了。

---

## 3. 先说你这张图最大的问题：主干和辅助关系画反了

你现在图里最容易让人误解的一点是：

**左边一大堆 prefix 信息经过 PRISM 流到右边 expert，看起来像 PRISM 是主干，expert 只是输出头。**

但你真实的方法不是这样。

真实逻辑应该是：

- 主干一直都是当前时刻的 policy 输入，也就是当前 observation
- PRISM 只是额外提供历史条件

你可以把它想成开车：

- 当前 observation 是“眼前路况”
- PRISM memory 是“我之前已经走过哪条路、前面发生过什么”
- 驾驶动作必须同时依赖这两者

所以改图时最重要的一件事是：

## 一定要把“当前 observation 直接进 action expert”的主路径画出来

建议你这样画：

- 左边新增一个清楚的大框：`Current Observation`
- 框里可以写：`current images o_t + current state s_t`
- 从这个框拉一条**粗一些的主箭头**直接进右边的 `Action Expert`

同时，PRISM 那一支不要画成唯一输入，而要画成旁路条件：

- `PRISM Context`
- `conditioning`
- `extra tokens / adapter`

也就是说，右边 expert 最好有两路输入：

1. 当前 observation 主输入
2. PRISM 历史条件输入

这一步一改，整张图的逻辑会立刻顺很多。

---

## 4. 第二个大问题：signature 的来源要画干净

你现在图里左侧有 `Robot State Prefix`，也有 `Visual Prefix`，而中间那个 signature 模块又画得比较像“把这些东西一起揉进去再折射出来”。

这会让人以为：

**signature 是从视觉前缀和状态前缀一起算出来的。**

但你论文里的定义不是这样。

你的定义是：

- signature 只来自 **state prefix**
- visual 不参与 signature 的计算
- visual 是在 memory write 的时候才跟 signature 合流

通俗讲就是：

- signature 回答的是：“机器人到目前为止走了一条什么样的状态轨迹？”
- visual 回答的是：“当前这一刻我看到了什么？”

这两件事不能在图上糊成一个模块。

所以你改图时，左边一定要拆成两条非常清楚的线：

### 线 A：状态前缀线

`Executed State Prefix`  
→ `State-Prefix Signature`  
→ `g_t, Delta g_t`

### 线 B：当前观测线

`Current Observation`  
→ `Visual / State Encoders`  
→ `v_t, p_t`

注意：

- 线 A 是“前缀摘要”
- 线 B 是“当前时刻特征”

这两条线之后再在 memory update 处汇合。

如果你图上还是让 visual prefix 直接扎进 signature 那个大盒子里，这张图就还是不够准确。

---

## 5. 第三个大问题：不要把 PRISM 画成“历史图片仓库”

你现在 memory bank 里放了一堆像小缩略图一样的东西，还配了类似 frame / experiment frame 的感觉。

这会让读者第一反应是：

- 这里面存的是历史图像
- 当前时刻会去检索过去的某几帧
- 这像一个 episode frame bank

但 PRISM 不是这个意思。

PRISM 的 memory 是：

- 固定大小
- latent slot
- 是神经网络内部状态
- 不是原始图片，也不是原始视频帧

你可以把它理解成：

**几个记忆格子，每个格子里存的不是照片，而是“压缩后的历史痕迹”。**

所以你改图时，memory bank 最好不要再画成“照片墙”。

建议改成：

- 若干个纯色或抽象的圆角矩形
- 每个格子里写 `m^(1)`, `m^(2)`, `m^(3)` ...
- 或者写 `slot 1`, `slot 2`, `slot 3`

总之要让读者一眼感觉：

**这是 latent memory slots，不是历史帧列表。**

标题也建议直接写：

- `Indexed Slot Memory`
- `Fixed-Budget Slot Memory`
- `Latent Memory Slots`

不要写：

- frames
- snapshots
- experience frames
- retrieved images

这些都会把方法带偏。

---

## 6. 第四个大问题：signature 不只是决定“写到哪”，还决定“怎么写”

你现在图里最显眼的是 `Routing Controller -> alpha -> slot memory`。

这当然有一部分是对的，因为 PRISM 确实有 routing weight。

但如果图只剩下这个，会让读者误解成：

- signature 只是一个索引器
- 它只负责选 slot
- 真正写入的内容主要还是视觉特征自己决定

而你的方法比这更强。

更通俗地说，signature 在 PRISM 里做了三件事：

1. 决定更偏向写哪个 slot
2. 参与生成这一步的写入内容
3. 参与控制这一步写入有多强

所以你不能只画一个 routing controller 就结束。

你更应该把中间画成：

### 一个“signature-conditioned memory update”模块

里面至少包括三部分含义：

- `Routing`：决定偏向写哪个槽
- `Write Proposal`：决定准备写什么内容
- `Gate`：决定这次写入有多大

如果你不想画太复杂，最简单也要做到：

- 在 memory bank 之前，不是只画 `alpha`
- 还要画一个 `Shared Write Proposal`
- 最好再画一个 `Gate`

这样读者才会知道：

**signature 不只是“给地址”，而是在参与整个更新过程。**

---

## 7. 你应该怎么理解“写入提案”和“门控”

如果你觉得这些词抽象，可以用很直白的话理解：

### 写入提案（write proposal）

意思是：

> “根据当前看到的东西、当前状态、以及历史摘要，我这一步准备往记忆里写一点什么。”

### 门控（gate）

意思是：

> “这次到底写多大？是轻轻改一下，还是明显覆盖一下？”

### 路由权重（routing weight）

意思是：

> “这些变化更应该写进哪个 slot？”

所以 memory update 的直觉可以理解成：

> 先想好写什么，再决定写去哪里，再决定写多重。

如果你想把图画得更友好，可以在模块旁边加一句小白话注释：

- `where to write`
- `what to write`
- `how strongly to write`

这比单独丢三个符号更容易读懂。

---

## 8. 第五个大问题：要把“当前”与“前缀”分开画

你现在图里容易让人混淆的一点是：

- 到底哪些东西是“完整前缀”
- 哪些东西是“当前时刻输入”

这在 PRISM 里非常关键。

因为 PRISM 并不是每一时刻都把整个视觉前缀重新喂进去做一次大计算。

它更像是：

- 当前时刻拿到 \(o_t, s_t\)
- 结合当前之前已经维护好的 memory state
- 做一次在线更新

也就是说，真正在线流动的是：

- 当前输入
- 递推状态

而不是：

- 整段历史原样缓存后反复重算

所以你的图里应该明确区分两种东西。

### 应该画成“前缀摘要”的

- `Executed State Prefix`
- `State-Prefix Signature`

### 应该画成“当前输入”的

- `Current Observation`
- `Current Visual Embedding`
- `Current State Embedding`

### 应该画成“递推状态”的

- `M_{t-1}`
- `M_t`

只要这三类分清楚，读者就不容易误解你的在线更新机制。

---

## 9. 第六个大问题：读出部分不要混成一团

你方法里，从 memory 到 expert 中间其实还有一步：

**memory readout + adapter**

这一步的作用是：

- 把记忆槽整理成 expert 能用的形式
- 不同 expert 可以有不同 adapter

如果这部分在图里画得太糊，读者就会不知道：

- 是所有 slot 全都直接送进 expert？
- 还是先池化成一个向量？
- signature token 和 memory token 是不是一回事？

所以建议你把从 memory 出来的东西拆成两路：

### 路 1：slot tokens

意思是：

> 各个记忆槽本身就可以作为 token 给 expert

### 路 2：pooled readout

意思是：

> 再从所有 slot 里读一个汇总向量

然后再把这些和 signature token 一起送进 `Context Adapter`。

你可以简单画成：

`Slot Memory`  
→ `Slot Tokens`

`Slot Memory`  
→ `Pooled Readout`

`Signature Tokens + Slot Tokens + Pooled Readout`  
→ `Context Adapter`

这一步画清楚之后，右侧会比现在更有层次。

---

## 10. 第七个大问题：右侧不要画得像多个 policy 同时运行

你现在右边在 `Action Expert` 周围又标了 diffusion / autoregressive / recurrent 这些。

你的本意是想说：

**PRISM 是 expert-agnostic。**

这个想法是对的。

但如果画法不当，读者会以为：

- 这些 policy 是并排存在的
- PRISM 会同时接多个 head
- 这是个多头系统

你真正想说的应该是：

> PRISM 先产生一个通用的历史条件接口，然后 ACT、diffusion、autoregressive、RNN 都可以各自接这个接口。

最稳的画法是：

- 右边只保留一个主框：`Action Expert`
- 在旁边加一句小字：`e.g., ACT / diffusion / autoregressive / RNN`

不要把它们都画成并行大模块。

这样更像“举例”，不容易被读成“并联结构”。

---

## 11. 四棱锥到底该不该留

短答案：

- **可以留作视觉隐喻**
- **不适合承担严格算法语义**

为什么？

因为四棱锥给人的直觉更像：

- 折射
- 分解
- 多种颜色分流
- 一个复杂输入被拆成多股信号

但你这里的 signature 模块真正做的事不是“视觉分光”，而是：

**从状态前缀里计算一个路径摘要。**

所以如果你把四棱锥放在 `Signature` 模块正中央，还让视觉和状态都流进去，读者会不自觉地以为：

- 这是一个 joint multimodal fusion / decomposition block

这就偏了。

最好的处理方式是二选一：

### 方案 A：把四棱锥当成 PRISM 的装饰性 logo

做法：

- 放在图中央偏标题感的位置
- 表示“这是 PRISM”
- 不承担具体计算含义

这种用法是安全的。

### 方案 B：不用四棱锥表达具体模块

做法：

- 直接把那个模块画成普通方框
- 写 `State-Prefix Signature`
- 或写 `Truncated Path Signature`

这种用法最学术、最稳。

### 不建议的用法

- 让视觉前缀和状态前缀都射进四棱锥
- 再从四棱锥里分出不同彩色信号

因为这会看起来像“多模态融合折射器”，不是你的方法定义。

---

## 12. 你可以直接照着下面这个顺序重画

如果你想少想一些，最简单就是按这个从左到右重画。

### 第 1 列：输入

上面放：

- `Executed State Prefix`

下面放：

- `Current Observation`

当前 observation 里如果你想细分，可以写：

- `current images o_t`
- `current state s_t`

### 第 2 列：两个不同来源的特征

从 `Executed State Prefix` 画到：

- `State-Prefix Signature`
- 输出 `g_t, Delta g_t`

从 `Current Observation` 画到：

- `Visual / State Encoders`
- 输出 `v_t, p_t`

### 第 3 列：核心 memory update

画一个大框：

- `Signature-Conditioned Slot Memory Update`

框里或框边标三件事：

- `Routing`
- `Write Proposal`
- `Gate`

这个框的输入是：

- `g_t, Delta g_t`
- `v_t, p_t`
- `M_{t-1}`

输出是：

- `M_t`

### 第 4 列：readout 和 adapter

从 `M_t` 出来画两路：

- `Slot Tokens`
- `Pooled Readout`

再把 `g_t, Delta g_t` 也接过来，汇总到：

- `Context Adapter`

输出：

- `PRISM Context`

### 第 5 列：action expert

右边放：

- `Action Expert`

它有两路输入：

1. 从左下角 `Current Observation` 直接过来的主路径
2. 从 `Context Adapter` 过来的 PRISM 条件路径

最后输出：

- `Action Chunk`

### 底部加一个回环

从 `M_t` 拉一条回环，标：

- `recurrent memory state`
- `used at next step`

这样读者就知道这是真正在线维护的状态。

---

## 13. 如果你只记住三条，记这三条就够了

### 第一条

`Current Observation` 一定要直接进 `Action Expert`。

这是主干。

### 第二条

`Signature` 只来自 `State Prefix`，不要让 `Visual Prefix` 进它。

### 第三条

`Slot Memory` 要画得像 latent slots，不要画得像历史照片库。

只要这三条改对，图就已经比现在准确很多了。

---

## 14. 最后给你一个“肉眼检查清单”

你改完图之后，自己问自己下面几个问题：

1. 读者会不会以为 PRISM 取代了 action expert？
2. 读者会不会以为 signature 是从视觉和状态一起算的？
3. 读者会不会以为 memory 里存的是历史图片帧？
4. 读者会不会看不出 current observation 仍然是决策主输入？
5. 读者会不会以为右边多个 policy 是同时运行的？

如果这 5 个问题你的答案都是“不会”，那图基本就稳了。

---

## 15. 最推荐的最终图意

你最想让读者一眼读出来的句子，其实应该是：

> PRISM uses a state-prefix signature to organize how current visual-state information is written into a fixed-budget slot memory, and then exposes the resulting history summary as a condition for an otherwise unchanged action expert.

翻成更通俗的话就是：

> PRISM 用状态历史来指导“当前看到的信息该怎么写进记忆”，再把这份记忆作为条件提供给原本的动作专家。

如果你的图能自然表达出这句话，就成功了。
