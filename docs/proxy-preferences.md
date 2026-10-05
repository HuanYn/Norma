# 初始偏好代理标注：2026-10-05

这批标注按用户授权，由助手直接查看公开照片后给出审美取舍。产品可以把它当成用户授权的初始偏好代理，但科研记录必须保留 `assistant_proxy` 来源，不能写成“用户亲手点选”或“真人金标准”。

## 本次交付与边界

- 8 对上下文偏好：6 对有选择，2 对平局；涉及 15 张现有 Wikimedia 公开照片。
- 每对保留任务、左右图、选中/未选中图、主观置信度、理由、局限、用户授权和规则版本。
- 标注之前已通过本地看图工具检查图像，不是按文件名或检索词自动赋标签；未调用云端模型，未上传私人照片。
- “未选中”不等于图片低质量；同一张照片可在不同任务中有不同取舍。平局没有被强行变成胜负。
- JSON 数据与公开源清单都锁定 SHA-256；15 张本地图像逐字节校验通过。
- 复用了历史受控实验图像：**整个源清单的 72 张图不再适合作为这批代理引导后的独立留出测试**。旧报告仍是旧协议的历史结果，不能改写成当前个性化效果。
- 未训练任何模型，未写入真人反馈表，未改变当前网页推荐。代理数据的个性化消费仍需后续接入与独立评估。

## 标注标准

先满足当前任务，再比较主体、层次、引导线和色彩。初始审美偏向电影感与较简洁的构图，但不把“安静”“冷色”变成所有任务的一刀切规则。缺少足够偏好信息时保留平局。集合多样性要在完整选片结果中检验，不能靠单张或单对标注证明。

本次不使用人物特写，因此没有标注“自然表情”，也没有猜测任何人的身份、关系或心理状态。每个 confidence 只是助手对自己判断稳定性的自评，不是校准概率。

## 如何使用

在项目根目录运行，默认只核验，不向数据库写入：

```powershell
python scripts/seed_proxy_preferences.py
```

显式导出开发用 JSONL，目标文件必须不存在：

```powershell
python scripts/seed_proxy_preferences.py --allow-assistant-proxy --output .norma/proxy-preferences-custom.jsonl
```

程序调用需显式传 `allow_assistant_proxy=True`。每条导出仍包含完整来源、两张图片内容哈希和开发集限制，不虚构用户 ID、相册 ID 或学习特征。未来独立评估应调用 `assert_held_out_disjoint` 检查图像哈希；该函数目前只是可复用防护，不声称已自动改接所有历史实验脚本。更换文件编码、字节或源清单时应显式发布新版本，不能静默覆盖已有标签。

## 实际核验

2026-10-05：28 个该模块测试通过；公开源清单哈希与 15 张图像内容哈希全部通过。JSONL 已生成在忽略目录 `.norma/proxy-preferences-20261005.jsonl`。这些是数据完整性/软件行为检查，**不是审美一致率、选片提升或用户满意度结果**。

数据集文件 SHA-256：`e09be7b6d564f46363a391ea2ced067e4283fd34233f853a686afa8ccdb1a353`。

## 逐对看图与理由

以下是标注依据，不是修复前后效果图；没有把代理判断冒充模型优化结果。图片引用本地已有公开演示相册，不复制原图进 Git。

### P1：选择 004

任务：旅行建筑开场照，偏电影感、空间层次清楚。

A：`003_97b4ada6b34a.jpg`

![P1 A](../.norma/demo-album-eval/003_97b4ada6b34a.jpg)

B：`004_9a3b0cbbe3ff.jpg`

![P1 B](../.norma/demo-album-eval/004_9a3b0cbbe3ff.jpg)

理由：004 的广场、两侧立面和钟塔构成清楚的纵深，冷灰天空与暖色石墙比较统一；003 的拱门也有框景，但近处石墙和树叶占比大，主体教堂部分被遮挡。

局限：在这对照片与这个任务下更偏好完整空间，不表示所有广角都优于近景。 主观置信度：0.8。

### P2：平局

任务：旅行建筑照片，保留地点特点和生活感。

A：`005_1839d30785cb.jpg`

![P2 A](../.norma/demo-album-eval/005_1839d30785cb.jpg)

B：`063_8707c5776037.jpg`

![P2 B](../.norma/demo-album-eval/063_8707c5776037.jpg)

理由：005 的彩色立面、老车和街上行人更有街头生活感；063 的木屋、桥和山体层次更完整。两张表现的是不同旅行体验，在没有城市或乡村偏好的前提下不强行分胜负。

局限：tie 保留，不转换成二元训练正负例；不推断图中人物身份或心情。 主观置信度：0.55。

### P3：选择 009

任务：挑一张夜景建筑照片，主体清楚、灯光与水面形成层次。

A：`009_490655e9e2e7.jpg`

![P3 A](../.norma/demo-album-eval/009_490655e9e2e7.jpg)

B：`023_53eabe29deee.jpg`

![P3 B](../.norma/demo-album-eval/023_53eabe29deee.jpg)

理由：009 能清楚看到高楼、低层建筑、摩天轮和倒影，主体尺度适合普通相册浏览；023 是很窄的全景，城市被压缩为密集灯点，作为单张主图细节和焦点较弱。

局限：不是认定全景质量差；超宽画幅展示任务下偏好可能反转。 主观置信度：0.88。

### P4：选择 027

任务：挑一张桥梁夜景，偏冷静、克制、有引导线。

A：`027_0fe667ea27f9.jpg`

![P4 A](../.norma/demo-album-eval/027_0fe667ea27f9.jpg)

B：`031_652aa3111696.jpg`

![P4 B](../.norma/demo-album-eval/031_652aa3111696.jpg)

理由：027 的桥索向远处延伸，蓝紫天空与偏冷水面形成较简洁的色调；031 的斜桥也有引导线，但大面积暖黄灯光与背景建筑让画面更热闹。本次任务偏向前者。

局限：这是任务相关的冷色审美代理，不是曝光或白平衡正确性的客观评分。 主观置信度：0.74。

### P5：选择 057

任务：选一张有电影氛围的山水主图，优先明确主体与前后景。

A：`056_eb029c859755.jpg`

![P5 A](../.norma/demo-album-eval/056_eb029c859755.jpg)

B：`057_bcb5c3f99d47.jpg`

![P5 B](../.norma/demo-album-eval/057_bcb5c3f99d47.jpg)

理由：057 的独立山峰、倒影和近处岸边形成明确前中后景，色彩范围较集中；056 的湖水和林木更明亮清新，但大片高饱和绿色分散了山体的主导性。

局限：不推断原始调色步骤；色彩评价仅依据可见图像。 主观置信度：0.78。

### P6：选择 058

任务：选一张有层次、引导线与独特构图的山水照片。

A：`058_8f397f811d13.jpg`

![P6 A](../.norma/demo-album-eval/058_8f397f811d13.jpg)

B：`059_8ac9419f4801.jpg`

![P6 B](../.norma/demo-album-eval/059_8ac9419f4801.jpg)

理由：058 用桥洞形成天然框景，溪流和石块把目光引向远处雪山；059 有好看的明暗变化，但右侧枝条和大块天空使边缘信息较多。本次更偏好 058 的集中构图。

局限：两张都有可用价值，不把未选照片标成低质量。 主观置信度：0.77。

### P7：选择 060

任务：选一张安静、简洁的山间居住场景。

A：`060_6f6d80358091.jpg`

![P7 A](../.norma/demo-album-eval/060_6f6d80358091.jpg)

B：`063_8707c5776037.jpg`

![P7 B](../.norma/demo-album-eval/063_8707c5776037.jpg)

理由：060 的小木屋落在亮草坡上，周围留出森林与天空，视觉节奏更安静；063 的村屋、桥、花草和山体信息更多，更适合讲村落细节。本次简洁要求下偏好 060。

局限：不将这一对推广为普遍的少建筑偏好；与第 2 对的任务不同。 主观置信度：0.76。

### P8：平局

任务：选一张横向山水照片，允许壮阔或宁静的情绪。

A：`061_34bf86f3afc9.jpg`

![P8 A](../.norma/demo-album-eval/061_34bf86f3afc9.jpg)

B：`062_9d19a272280c.jpg`

![P8 B](../.norma/demo-album-eval/062_9d19a272280c.jpg)

理由：061 有悬崖、峡湾和远处水道的尺度感；062 有雾气、水面倒影和近处透明水底的细节。两张都满足横向山水任务，但情绪不同，没有足够依据作稳定取舍。

局限：tie 保留，集合多样性要在完整候选集合层面另行检验，不能由这对标签证明。 主观置信度：0.6。

## 来源与许可

以下信息来自仓库中锁定的公开源清单；如对外再分发照片，应再次核对原页面、作者署名与各自许可条款。本轮仓库产物只含标注、来源链接和程序，不包含图片文件，尚未提交。

- [003_97b4ada6b34a.jpg](https://commons.wikimedia.org/wiki/File:Architecture_Antiquity_Old_Travel_Arch_Monastery.jpg) — fietzfotos，CC0。
- [004_9a3b0cbbe3ff.jpg](https://commons.wikimedia.org/wiki/File:Piazza_Del_Duomo_Lecce_Italy_Travel_Photography_%28171395721%29.jpeg) — Giuseppe Milo，CC BY 3.0。
- [005_1839d30785cb.jpg](https://commons.wikimedia.org/wiki/File:Architecture_Travel_City_Street_Tourism_Cuba.jpg) — SweetMellowChill，CC0。
- [009_490655e9e2e7.jpg](https://commons.wikimedia.org/wiki/File:Sydney_%28AU%29%2C_Darling_Harbour_--_2019_--_3193-5.jpg) — Dietmar Rabich，CC BY-SA 4.0。
- [023_53eabe29deee.jpg](https://commons.wikimedia.org/wiki/File:Wellington_City_Night.jpg) — Donovan Govan.，CC BY-SA 3.0。
- [027_0fe667ea27f9.jpg](https://commons.wikimedia.org/wiki/File:Pont_de_Brooklyn_de_nuit_-_Octobre_2008_edit.jpg) — Martin St-Amant (S23678)，CC BY 3.0。
- [031_652aa3111696.jpg](https://commons.wikimedia.org/wiki/File:Sz%C3%A9chenyi_Chain_Bridge_in_Budapest_at_night.jpg) — Wilfredor，CC0。
- [056_eb029c859755.jpg](https://commons.wikimedia.org/wiki/File:Untersberg_Mountain_Salzburg_Austria_Landscape_Photography_%28256594075%29.jpeg) — Giuseppe Milo，CC BY 3.0。
- [057_bcb5c3f99d47.jpg](https://commons.wikimedia.org/wiki/File:Kirkjufell_Mountain_Iceland_Travel_Photography_%28227214943%29.jpeg) — Giuseppe Milo，CC BY 3.0。
- [058_8f397f811d13.jpg](https://commons.wikimedia.org/wiki/File:Sligachan_Scotland_Landscape_Travel_Photography_%28121036503%29.jpeg) — Giuseppe Milo，CC BY 3.0。
- [059_8ac9419f4801.jpg](https://commons.wikimedia.org/wiki/File:The_Highlands_Scotland_Travel_Landscape_Photography_%28123275593%29.jpeg) — Giuseppe Milo，CC BY 3.0。
- [060_6f6d80358091.jpg](https://commons.wikimedia.org/wiki/File:The_House_Alta_Badia_Italy_Travel_Landscape_Photography_%28122696751%29.jpeg) — Giuseppe Milo，CC BY 3.0。
- [061_34bf86f3afc9.jpg](https://commons.wikimedia.org/wiki/File:Lysefjord_Norway_Landscape_Travel_Photography_%28122226717%29.jpeg) — Giuseppe Milo，CC BY 3.0。
- [062_9d19a272280c.jpg](https://commons.wikimedia.org/wiki/File:Myrkdalsvatnet_Myrkdalen_Norway_Landscape_Travel_Photography_%28120472613%29.jpeg) — Giuseppe Milo，CC BY 3.0。
- [063_8707c5776037.jpg](https://commons.wikimedia.org/wiki/File:San_Cassiano_Alta_Badia_Italy_Travel_Landscape_Photography_%28131834241%29.jpeg) — Giuseppe Milo，CC BY 3.0。

## 面试说明

可以说：“我做了带上下文和来源标记的冷启动偏好代理数据，保留平局和判断理由，用内容哈希防止图像漂移，并将已暴露的公开源图排除出后续独立留出评测。”不能说：“8 条真人反馈已经验证个性化提升”或者“已经进行了 DPO”。这次没有参数更新；后续可用冻结模型加偏好记忆做条件化选片，再用未见过的照片与独立评价检验效果。

一句话总结：**把授权代理偏好做成可追溯、可验证、与直接用户反馈及评测集区分的数据，供下一步无需训练的个性化重排使用。**
