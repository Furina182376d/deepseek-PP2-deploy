# Phase 3 — 节点间通信加密

## 阶段目标

在 PP=2 两节点部署中，**加密所有跨节点的张量通信**（PP 激活数据），
防止明文数据在 eth0 链路上被截获。假设密钥已通过带外方式协商完成。

提供两条路径：

| 路径 | 原理 | 改动范围 | 性能影响 |
|------|------|----------|----------|
| **A — `comm_crypto.py`**（推荐首选） | 在 PyTorch `dist.send/recv` 层插入 AES-256-GCM 加密 hook，仅加密跨节点 P2P 流量 | vLLM 零改动；仅 monkey-patch | 需要 CPU-GPU 数据搬运，预期 PP 边界额外 2-5ms |
| **B — `setup_ipsec.sh`**（备选/配合） | OS 内核级 IPsec transport mode，加密两节点间全部 IP 流量 | **零代码改动**，透明于 vLLM | 极小（内核 ESP 硬件卸载） |

两条路径可**独立使用，也可叠加**。

## 文件说明

| 文件 | 作用 |
|------|------|
| `comm_crypto.py` | 加密通信核心模块 — 提供三个层次：① 底层原语 `encrypt_tensor()` / `decrypt_tensor()`（AES-256-GCM 加密 tensor 原始字节，若 `cryptography` 未安装则降级为 XOR fallback 并警告）；② `EncryptedP2PGroup` 类包装 `ProcessGroup`，`send/recv` 时对跨节点 rank 自动加解密；③ `install_encrypted_pp_hooks()` 全局 monkey-patch `torch.distributed.send/recv`，使所有跨节点 P2P 调用自动加密。密钥从环境变量 `VLLM_COMM_PSK` 注入 |
| `setup_ipsec.sh` | IPsec transport mode 配置脚本 — 在两节点 eth0 之间建立 ESP 加密隧道（AES-128 + HMAC-SHA256）。需 **root 权限**运行。脚本自动生成 SA 和 policy 规则，覆盖 192.168.0.63 ↔ 192.168.0.65 的全部 IP 流量。部署后可用 `tcpdump -i eth0 esp` 验证只有 ESP 密文帧 |

## 使用方式

### 路径 A：应用层加密（推荐）

```bash
# 1. 安装依赖（用户执行）
pip install cryptography

# 2. 生成并共享密钥
export VLLM_COMM_PSK="$(openssl rand -base64 32)"
# 将此值在两节点设为相同的环境变量

# 3. 正常启动（profile_dsv4_pp.py 在 vLLM 导入前自动安装 hooks）
cd phase1
VLLM_COMM_PSK="$VLLM_COMM_PSK" ./launch_pp.sh 0   # Node 0
VLLM_COMM_PSK="$VLLM_COMM_PSK" ./launch_pp.sh 1   # Node 1
```

### 路径 B：IPsec（备选/零代码改动）

```bash
# 1. 生成密钥（分别在两节点执行）
# 编辑 setup_ipsec.sh，将 AES_KEY 和 AUTH_KEY 替换为 openssl rand -hex 生成的随机值

# 2. 部署（需 root）
sudo ./setup_ipsec.sh 0   # Node 0
sudo ./setup_ipsec.sh 1   # Node 1

# 3. 验证
sudo ip xfrm state          # 检查 SA 已建立
sudo tcpdump -i eth0 esp    # 确认跨节点流量为 ESP 密文

# 4. 正常启动 PP 部署（无需任何代码改动）
cd phase1
./launch_pp.sh 0
```

### 验证加密生效

```bash
# 抓包对比：有加密时只能看到 ESP 帧（无明文 NCCL/torch 载荷）
sudo tcpdump -i eth0 host 192.168.0.65 -c 20 -X
```

## 文件依赖关系

```
comm_crypto.py ──> profile_dsv4_pp.py  （启动时自动 import + install hooks）
setup_ipsec.sh  ──> launch_pp.sh       （部署 IPsec 后正常启动即可，无需额外集成）
```

## 注意事项

- `comm_crypto.py` 仅加密跨节点流量；同节点内（PP stage 内部的 TP all-reduce）不加密（性能优化）
- 若 `cryptography` 未安装，会降级为 XOR 混淆（仅用于测试连通性，不提供实际安全性）
- IPsec transport mode 要求两节点间无中间 NAT/路由器做 IP 改写（同子网满足此条件）
- NCCL over TCP（`NCCL_IB_DISABLE=1`）是 IPsec 能透明加密的前提（已在 `launch_pp.sh` 中设定）
