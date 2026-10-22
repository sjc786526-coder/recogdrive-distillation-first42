# ReCogDrive：首版42候选固定教师的蒸馏复现与改进

给已有相同ReCogDrive、RL Base、NAVSIM数据／地图／评分缓存的合作者使用。目标是分析本轮H／T／HT结果，并修改训练方法提升学生能力。**固定使用已有首42候选的场景固定R2教师，不重选教师、不引入后来扩库的教师。**

代码：https://github.com/sjc786526-coder/recogdrive-distillation-first42  
权重与数据：https://huggingface.co/datasets/WHU-SJC/recogdrive-distillation-first42

**数据正在上传。必须以HF仓库最后的完整清单和SHA校验结果判断交付是否齐全。** 原始实验资产未移动、覆盖或重新量化。这里的适配和工程计算都发生在独立交付副本。

## 交付什么

| 位置 | 内容 |
|---|---|
| 本仓库`src/`、`train.py`、`launch.py` | 原科学Student／损失／DDP包装／预取器，以及独立路径、分片数据和训练调度适配 |
| `native/` | 本次实际使用的模型、预处理与原生PDMS评分代码快照；不依赖阿里DSW、DLC或CPFS |
| `configs/original/`、`reference/` | 原训练配方、来源SHA、参数清单、验证主权重与测试最佳诊断身份 |
| HF `checkpoints/` | H／T／HT各5个epoch端点，共15份，原始checkpoint约6.14GB |
| HF `teacher/` | 103288场景的固定R2完整FP32接口＋五条原始轨迹／评分，约414GB的无损分片 |
| HF `reference/` | 原Base与固定R2轨迹／七项评分、全部42候选评分，便于研究教师质量与学生拟合 |
| HF `results/`、`training_logs/` | 已验收的逐场景学生评价、原训练日志；未完成计算不会冒充最终结果 |

权重不是整套Base模型：checkpoint保存LoRA、两处投影的完整可训练参数、优化器和随机状态。加载必须配套本仓库Student结构和同SHA的RL Base。15个原始端点保留是为了研究学习过程；不需要复制原始图像、地图、Base和VLM大权重。

## 教师与数据定义

- **R2**：每个场景按S0～S4真实PDMS五次均分选一个实际分支，包含Base。没有“只有低于70才修”的门槛，也没有额外安全筛选。同一场景五次采用同一分支，轨迹与接口来自同次实际执行。
- 接口是动作头`feature_encoder`的输出，FP32 `[2800,384]`，**包括padding**。同一场景的五次采样接口完全相同，故只保存一份接口，配五条轨迹。不用只含valid-token的旧导出替代当前完整目标。
- 探索训练85813场景，其中实际梯度训练I／P为85339；E474不训练但评价。验证17475场景不参与本轮学生梯度。测试数据仅用于评价，不进入TeacherDataset梯度训练。
- 训练独立场景比例：难例修复1847（2.1643%）、其他修复47625（55.8068%）、保持35867（42.0288%）。自然比例，没有难例过采样。I只要求教师五次均分真实优于Base，不要求教师先到90分；保持组监督目标为原Base。
- 按完整验证17475场景S0～S4均分选择的主权重为H5、T5、HT5；S0～S4测试最佳诊断为H5、T5、HT2。测试最佳为事后诊断，不能代替验证主权重。原512面板历史选择H4／T4／HT2仅辅助保留。

## 实际训练配方

| 组别 | 接口MSE系数 | 轨迹损失系数 |
|---|---:|---:|
| H | 1 | 0 |
| T | 0 | 1 |
| HT | 1 | 80.21022962075446 |

接口损失是完整token、完整坐标MSE。轨迹损失为x/40、y/10后的SmoothL1，再加0.1倍航向余弦误差。HT权重按32个训练I样本的接口梯度量级校准，未用验证／测试定权，不代表轨迹“重要80倍”。

视觉24层、语言28层共292处线性模块用LoRA（rank16／alpha32／dropout0）；视觉→语言投影和VLM→动作接口投影全参，其余原参数及动作生成网络冻结。可训练34008448参数：LoRA24756224、视觉投影8662016、动作接口590208。后续原生5步DDIM保持可微，T梯度穿过冻结动作网络回传。

原实验每组16卡、microbatch1、累积4、global batch64，训练5轮／6670优化步，共426880次曝光（含185次尾部重复）。AdamW：LoRA LR5e-5、投影1e-4、betas(.9,.999)、eps1e-8、weight decay0、clip1；warmup200后余弦降至初始LR的10%。VLM原生BF16、动作FP32、可训练参数FP32、逐层非重入梯度检查点。这里只做一次初始化，不代表各方法已经完成超参充分搜索。

## 下载和空间

```bash
git clone https://github.com/sjc786526-coder/recogdrive-distillation-first42.git
cd recogdrive-distillation-first42

# 使用已有环境的huggingface_hub CLI；下载到最终目录，避免再手工复制一份
hf download WHU-SJC/recogdrive-distillation-first42 --repo-type dataset --local-dir /your/storage/first42-data
```

教师tar已经包着压缩NPZ，不需要再整批解压。数据读取器按索引从tar偏移直接读取原始NPZ字节并核对SHA，避免压缩包与解包目录各占一份。建议为交付数据留500～600GB，再另外留训练输出空间。可以先只下`manifests/*`、`reference/*`、`results/*`、`checkpoints/*`做结果分析；完整H／T／HT复训需要教师分片。

## 配本地路径，先检查和冒烟

```bash
cp configs/paths.example.json paths.local.json
# 填息壤实际checkpoint、VLM、maps、logs、sensor和metric缓存路径
python -B check_assets.py --paths paths.local.json --data /your/storage/first42-data
python -B check_assets.py --paths paths.local.json --data /your/storage/first42-data --fast --verify-shards

# 纯CPU分析已保存结果，无需模型或GPU
python -B analyze.py --data /your/storage/first42-data --out outputs/saved-results

# 先用原生RL Base做两场景实际推理与评分（不加载学生checkpoint就是零LoRA的Base）
CUDA_VISIBLE_DEVICES=0 python -B evaluate.py --paths paths.local.json --data /your/storage/first42-data \
  --cohort navtrain --subset validation --limit 2 --out outputs/base-smoke

# 加载HT第5轮；H／T的用法相同
CUDA_VISIBLE_DEVICES=0 python -B evaluate.py --paths paths.local.json --data /your/storage/first42-data \
  --group HT --checkpoint /your/storage/first42-data/checkpoints/HT/epoch05.pt \
  --cohort navtrain --subset validation --limit 2 --out outputs/ht5-smoke
```

RL Base checkpoint SHA256：`23fb949b10387efeff87e093703d5faf41b61b9e5ead16d4076661e3c7c08a20`。VLM文件SHA和源评分协议在`native/configs/repair_final_baseline.json`。已有“同名缓存”仍要先核对golden轨迹和评分；相同seed不能保证不同加速卡／Torch版本逐位相同。

## 重新训练和修改方法

```bash
# 单节点8卡示例，保留global batch64，自动累积8次；1／2／4／16卡同样可配置
python -B launch.py --paths paths.local.json --data /your/storage/first42-data \
  --group HT --devices 0,1,2,3,4,5,6,7 --out outputs/ht-reproduction

# 可用--overrides传新JSON调整lambda_H、lambda_T、学习率等；必须新输出目录
python -B launch.py --paths paths.local.json --data /your/storage/first42-data \
  --group HT --devices 0,1,2,3,4,5,6,7 --overrides configs/overrides.example.json --out outputs/ht-new-weight
```

改变卡数时保持global batch64及microbatch1，累积次数=64/总卡数；总卡数必须整除64。多节点使用相同可见卡数、共享输出目录，分别设置`--nodes N --node-rank R --master-addr IP --master-port PORT`。每个子进程只看到自己的单卡，避免所有rank误用同一卡。启动器只管理自己创建的Popen进程，没有DLC或云端操作。

默认仍使用原512开发面板作每epoch在线监测；训练后应另跑完整验证17475场景选择主权重。`evaluate.py`支持`--rank/--world-size`独立分片、`--noises`和逐场景断点续跑。训练与验证采用原S0～S4；测试原五组和S5～S9另列。复制来的原15个checkpoint可推理分析；其16rank随机状态不应直接当作不同卡数下的精确断点恢复。`--resume`用于同一新portable配置的断点。

## 原平台实际噪声、种子与验证边界

HF `noise/navtrain_noise.npy`保存所有训练／验证场景S0～S4的实际参考随机流（约297MB），`noise/navtest_noise.npy`保存测试S0～S9（约70MB）。其生成结果与历史保存golden噪声逐位核对。若这些文件存在，portable采样器默认重放原平台噪声，保持原生裁剪和采样计算；可用`RECOGDRIVE_NATIVE_RNG=1`显式切回本机按seed生成，并将该选择记录进训练配置。固定噪声仍不能消除A800与PPU之间的浮点计算差异。

动作种子为`int(sha256(f"{token}|20261008|{j}").hexdigest()[:8],16)`，S0～S4对应j=0…4，排除GPU／rank编号。完整随机流包含一份初始 `[1,8,3]` 噪声及五份逐步噪声，原生时间步 `[80,60,40,20,0]`、eta1。初始化2026102101、数据顺序2026102102、开发面板2026102105；五轮噪声轮换`(epoch + seed(token,0)%5)%5`。

`reference/original_code/`保留原科学代码，`src/`中的Student、损失、DDP包装和预取器按原字节复制。适配仅负责本地路径、原字节分片读取、配置与调度。本包已在原PPU平台对Base和三组学生进行真实一致性检查；**尚未在息壤A800及其多卡环境验收**，不能直接承诺跨平台逐位相同。先完成当地小规模轨迹、评分、梯度和多卡检查，再开始长训练。

模型和数据的文件身份、原始来源与验收详见`reference/`以及HF数据完成清单。教师Oracle分数不是学生能保证获得的分数；接口／轨迹拟合下降也不自动等于真实PDMS提升。
