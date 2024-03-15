# XTuner 微调命令

启动命令：
```bash
NPROC_PER_NODE=8 xtuner train /home/jovyan/mathripper/project/Internlm/full_7b/full_7b.py --deepspeed deepspeed_zero2 >output.log 2>&1
```

pth 模型文件转 LoRA 模型文件
```bash
xtuner convert pth_to_hf qlora_config.py ./work_dirs/qlora_config/iter_3210.pth ./hf
```

将 HuggingFace adapter 合并到大语言模型
```bash
xtuner convert merge /home/jovyan/mathripper/models/internlm2-chat-20b ./hf ./merged --max-shard-size 10GB --device cpu
```

运行模型
```bash
xtuner chat /home/jovyan/mathripper/models/internlm2-chat-20b --adapter ./hf --prompt-template internlm2_chat
```

