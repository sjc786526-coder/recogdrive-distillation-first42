# 在本地重新生成固定R2教师

此流程替代传输约414GB接口目标。只需下载约1.06GB：新增冻结参数及14个golden样例约510MB，其余为已有噪声、场景清单及参考评分／轨迹。模型和NAVSIM资产使用当地已有副本。**不重新拟合专家，不重新遍历42候选，不重新选择教师。**

## 先下载并核对，再做小规模检查

```bash
git pull --ff-only
python -B download_teacher_assets.py --data /your/storage/first42-reference
python -B prepare_teacher.py --data /your/storage/first42-reference
cp configs/paths.example.json paths.local.json
# 填入当地RL Base、VLM、地图、训练／测试日志、传感器和评分缓存路径
python -B check_assets.py --paths paths.local.json --models-only

CUDA_VISIBLE_DEVICES=0 python -B replay_teacher.py \
  --paths paths.local.json --data /your/storage/first42-reference \
  --out /your/storage/teacher-golden-check --golden-only
```

`download_teacher_assets.py`只下载固定的10个文件，逐个核对SHA256，不会下载`teacher/*.tar`。`prepare_teacher.py`核对压缩包、全部37份独立参数和14个golden样例，并按白名单安全解包。所有参数为原始字节，无量化。

新输出与参考下载目录必须分开。重放固定R2分支，分别运行当地Base和教师，使用相同参考噪声计算五条轨迹，并用实际PDMS重新评分。目标接口与教师轨迹来自同次执行，完整包含padding；同一场景五次接口须完全相同。保留原始I／P／E划分，不因当地重放分数变化自动改选分支或重划数据。

14个golden覆盖实际获选的不同干预位置、语义token组、内部观测门控与Base。每个`teacher_cases/<token>/RESULT.json`记录轨迹最大误差、接口相对L2、教师及Base评分差异。输入token、mask和position必须一致。这里没有把跨硬件数值差异伪装成精确重现：**原PPU平台14场景的接口、五轨迹及评分已实际逐位重放通过；A800和多卡需要在当地先验收。** 同平台可加`--strict-original`要求严格相等；A800先看差异，不应盲目加大容差或沿用旧评分。明显差异应先排查模型、输入和数值计算，再投入全量。

## 全量采集与CPU验收

```bash
# 独立新目录，单节点8卡示例；卡数和节点数可配置
python -B launch_teacher.py --paths paths.local.json \
  --data /your/storage/first42-reference --out /your/storage/first42-local \
  --devices 0,1,2,3,4,5,6,7

# 所有节点／rank完成后，在CPU上汇总
python -B finish_teacher.py --out /your/storage/first42-local
python -B check_assets.py --paths paths.local.json --data /your/storage/first42-local
```

多节点分别加`--nodes N --node-rank R`，各节点使用相同卡数、共享输出目录。教师rank互相独立，不需要NCCL；种子不含rank或GPU编号。每个进程只看到一张卡。按场景边界断点：相同参数重新运行同一命令会校验并跳过已完成场景；修改代码、候选、噪声、划分或rank分配后使用新目录。启动器只管理自身创建的进程。

原模型每场景只运行一次Base VLM，然后继续所选修复分支；五个噪声共享接口。输入预取、评分与下一场景推理重叠；完全相同的轨迹由原评分器去重。先按每卡一个进程开始，实测当地吞吐后再决定是否增加并发。不要照搬原平台并发结论。

全103288场景生成后，`finish_teacher.py`检查覆盖、冻结身份、同次执行标记并生成本地`TEACHER_INDEX.json`、本地重新评分的Base／教师数值表和`TEACHER_READY.json`。可加`--verify-files`全量重算NPZ哈希，代价是读一遍全部接口文件。数据仍需约420GB存储，采用逐场景压缩NPZ，不需要tar再解包占两份。

这里只生成训练／验证侧教师。测试不用于梯度；测试学生由现有`evaluate.py`从原始观测独立推理。若要在A800报告相对Base的测试增益，应在同一当地环境重跑Base测试，不能把不同平台的Base分数混在一起。

## 直接接现有蒸馏

```bash
python -B launch.py --paths paths.local.json --data /your/storage/first42-local \
  --group HT --devices 0,1,2,3,4,5,6,7 --out outputs/ht-a800-reproduction
```

H、T同理，均从RL Base重新初始化。原H／T／HT权重、LoRA和全参范围、数据分组、global batch64及训练预算继续按原配方。训练配置绑定本地教师索引与重放身份的SHA，避免断点时悄悄混入另一版教师。局部工程数据带`ENGINEERING_ONLY.json`，正式训练会拒绝它。主权重由完整验证集选择；原平台成绩与A800重放／改进成绩分开报告。
