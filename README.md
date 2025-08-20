# 🚀 Agent101: AI Agent实战系列

<div align="center">

![AI Agent](https://img.shields.io/badge/AI%20Agent-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10+-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-red.svg)
![License](https://img.shields.io/badge/License-MIT-orange.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)

**🤖 从零开始构建智能AI Agent系统！**<br/>

[📖 课程内容](#-课程内容) • [🚀 环境搭建](#-环境搭建) • [🎯 实战项目](#-实战项目) • [🛠️ 技术栈](#️-技术栈)

</div>

---

## 📖 课程内容

本课程专注于AI Agent的实战开发，采用循序渐进的学习方式，从基础环境搭建到企业级应用部署，每个模块都包含完整的理论讲解和实战练习。通过本课程，您将掌握构建智能AI Agent系统的核心技能。

### 🎯 课程特色

- **实战导向**：每个模块都有完整的实战项目和代码示例
- **企业级应用**：涵盖真实的企业应用场景和最佳实践
- **技术前沿**：涵盖最新的AI Agent技术和框架
- **完整生态**：从环境搭建到生产部署的完整技术栈

### 📚 课程大纲

| 模块 | 章节标题 | 核心技术 | 实战项目 |
|------|----------|----------|----------|
| **模块一** | AI Agent环境搭建与基础配置 | Python环境、开发工具、基础配置 | 环境配置与第一个LLM应用 |
| **模块二** | 提示词工程与上下文管理 | Prompt Engineering、Context Engineering | 高效提示词设计与上下文优化 |
| **模块三** | 大模型部署与推理优化 | 模型部署、性能优化、推理加速 | 本地模型部署与性能测试 |
| **模块四** | MCP协议与工具集成 | MCP协议、工具调用、API集成 | AI旅行规划MCP智能体 |
| **模块五** | 多角色Agent协作系统 | LangChain、AutoGen、Ango | 多Agent协作与角色分工 |
| **模块六** | 工作流自动化与n8n集成 | 可视化工作流、API集成、自动化 | 智能工作流设计与部署 |
| **模块七** | RAG检索增强生成系统 | RAG架构、向量数据库、智能检索 | 企业文档智能问答系统 |
| **模块八** | 模型微调与优化 | LoRA、Q-LoRA、性能优化 | 领域特定模型微调 |
| **模块九** | 企业级项目实战 | 全技术栈融合、生产部署 | 智能法律咨询助手 |

---

## 🛠️ 技术栈

### 🤖 AI Agent核心框架

| 技术分类 | 核心技术 | 主要工具/框架 |
|----------|----------|---------------|
| **Agent框架** | LangChain、AutoGen、Ango | 单Agent、多Agent协作、工具调用 |
| **提示词工程** | CoT、Self-Reflection、Few-shot | 上下文管理、提示词优化 |
| **工具集成** | Function Calling、API调用 | 自定义工具、第三方服务集成 |

### 🔍 检索增强生成 (RAG)

| 技术分类 | 核心技术 | 主要工具/框架 |
|----------|----------|---------------|
| **向量数据库** | 相似度检索、元数据过滤 | ChromaDB、Qdrant、Milvus |
| **嵌入模型** | 文本向量化、多语言支持 | BGE、E5、OpenAI Embeddings |
| **文档处理** | 分块、加载、预处理 | LangChain、Unstructured |
| **检索优化** | 重排序、混合检索 | BM25、Dense Retrieval |

### 🚀 模型部署与推理

| 技术分类 | 核心技术 | 主要工具/框架 |
|----------|----------|---------------|
| **模型推理** | VLLM、TGI、Ollama | PyTorch、Transformers、CUDA |
| **API调用** | OpenAI API、DeepSeek API | HTTP Client、异步请求 |
| **性能优化** | 量化、缓存、批处理 | INT8量化、KV缓存 |

### 🔧 工作流自动化

| 技术分类 | 核心技术 | 主要工具/框架 |
|----------|----------|---------------|
| **可视化编排** | 节点连接、数据流 | n8n、Zapier |
| **触发器** | Webhook、定时任务 | Cron、事件驱动 |
| **集成能力** | API连接、数据转换 | HTTP请求、数据映射 |

### 🎯 模型微调

| 技术分类 | 核心技术 | 主要工具/框架 |
|----------|----------|---------------|
| **高效微调** | LoRA、Q-LoRA、Adapter | PEFT、Unsloth、LlamaFactory |
| **数据处理** | 数据清洗、格式转换 | Datasets、Pandas |
| **模型评估** | 指标计算、A/B测试 | BLEU、Rouge、人工评估 |

### 🚀 部署运维

| 技术分类 | 核心技术 | 主要工具/框架 |
|----------|----------|---------------|
| **容器化** | Docker、Kubernetes | Docker Compose、K8s |
| **监控日志** | 性能监控、日志分析 | Grafana、ELK Stack |
| **负载均衡** | 高可用、弹性伸缩 | Nginx、云负载均衡 |
| **成本优化** | 模型量化、资源调度 | INT8量化、GPU调度 |

---

## 🚀 环境搭建

### 📖 Ubuntu系统配置手册

在开始环境搭建前，建议先阅读我们提供的详细配置手册：

📚 **[Ubuntu 22.04.4 运维配置手册](00-agent-env/linux_ops/Ubuntu22.04.4-运维配置手册.md)**

该手册面向Linux技术初学者，涵盖了从系统基础配置到开发环境搭建的完整流程，包括：
- 🔧 SSH工具配置和静态IP设置
- 👤 用户权限管理和系统时间同步
- 💾 磁盘管理和DNS配置
- 🔨 Git环境、Docker、Node.js等开发工具安装
- 📊 系统监控与维护最佳实践

### 💻 系统要求

- **操作系统**: Ubuntu 22.04 LTS (推荐)
- **Python**: 3.10.18
- **CPU**: >= 2 C
- **内存**: >= 16GB RAM
- **存储**: >= 100GB 可用空间
- **GPU**: NVIDIA GPU (可选，推荐用于模型微调和推理加速)

### 必要的API Keys
在开始之前，请准备以下API Keys中的至少一个：
- [OpenAI API Key(官方)](https://platform.openai.com/api-keys)
- [OpenAI API Key(国内代理)](https://www.apiyi.com/register/?aff_code=we80)
- [DeepSeek API Key](https://platform.deepseek.com/api-keys)
  
### 配置外网访问

> **适用系统：** Linux  
> **用途：** 确保顺利访问Google、HuggingFace、Docker Hub、GitHub等海外服务

#### 操作步骤

**1. 购买代理服务**
- 注册地址：https://yundong.xn--xhq8sm16c5ls.com/#/register?code=RQKCnEWf
- 选择适合的套餐完成购买

**2. 安装 V2rayA 客户端**
- 官方安装教程：https://v2raya.org/docs/prologue/installation/debian/
- 按照教程完成 V2rayA 的安装和配置

**3. 导入订阅链接**
- 获取订阅链接：https://yundong.xn--xhq8sm16c5ls.com/#/knowledge
- 在 V2rayA 界面中导入订阅

**4. 选择并启动节点**
- 在节点列表中选择延迟较低的节点
- 点击左上角的"启动"按钮激活代理

**5. 配置代理模式**
- 访问 V2rayA 管理界面：http://127.0.0.1:2017/
- 进入 **设置** → **透明代理/系统代理**
- 选择：**"分流规则与规则端口所选模式一致"**

#### 验证配置
```bash
# 测试外网连接
curl -I https://www.google.com
curl -I https://huggingface.co
```

#### 方法二：手动安装

```bash
# 1. Git安装(已安装请忽略)
## 更新包列表
sudo apt update
## 安装 Git 
sudo apt install git -y
## 验证Git是否成功安装
git --version

# 2. Git配置（PUSH需要）
## 配置用户名
git config --global user.name "Your Name"
## 配置用户邮箱
git config --global user.email "your.email@example.com"


# 3. 克隆项目
git clone https://github.com/FlyAIBox/Agent-101.git
cd Agent-101

# 4. 运行自动化配置脚本
chmod +x 00-agent-env/setup_agent101_dev.sh
./00-agent-env/setup_agent101_dev.sh

# 5. 激活环境
conda activate agent101
```


#### 1. GPU驱动与CUDA配置（可选/微调才会用到）

---

## 🎯 实战项目

### 🛫 AI旅行规划系统
- **位置**: `07-agent-rag/`
- **技术栈**: RAG、向量数据库、智能检索
- **功能**: 智能旅行规划、景点推荐、行程优化
- **特色**: 完整的RAG系统实现，包含文档处理、向量检索、智能问答

### 🤝 多角色Agent协作系统
- **位置**: `04-agent-multi-role/`
- **技术栈**: LangChain、AutoGen、Ango
- **功能**: 多Agent协作、角色分工、任务分配
- **特色**: 展示不同Agent框架的特点和适用场景

### 🔧 MCP智能体系统
- **位置**: `03-agent-mcp/`
- **技术栈**: MCP协议、工具调用、API集成
- **功能**: 工具集成、API调用、智能决策
- **特色**: 基于MCP协议的标准化工具集成方案

### 📚 企业文档智能问答
- **位置**: `07-agent-rag/`
- **技术栈**: RAG、ChromaDB、LangChain
- **功能**: 文档检索、智能问答、上下文优化
- **特色**: 企业级RAG系统的完整实现

### ⚖️ 智能法律咨询助手
- **位置**: `08-agent-project/`
- **技术栈**: 全技术栈融合、企业级架构
- **功能**: 法律条文检索、合同分析、风险评估
- **特色**: 企业级项目的完整实现和部署

### 🔄 工作流自动化系统
- **位置**: `05-agent-workflow-n8n/`
- **技术栈**: n8n、可视化工作流、API集成
- **功能**: 业务流程自动化、数据集成、智能决策
- **特色**: 无代码工作流设计与AI Agent集成

---

## 📋 学习路径建议

### 🎓 学习阶段

1. **基础阶段** (模块一~二): 环境搭建 → 提示词工程 → 上下文管理
2. **进阶阶段** (模块三~五): 模型部署 → MCP集成 → 多Agent协作
3. **高级阶段** (模块六~八): 工作流自动化 → RAG系统 → 模型微调
4. **项目阶段** (模块九): 企业级项目设计与部署

### 🎯 学习成果

完成本课程后，您将能够：

- ✅ 独立搭建AI Agent开发环境
- ✅ 设计高效的提示词和上下文管理系统
- ✅ 部署高性能的大模型推理服务
- ✅ 构建企业级RAG问答系统
- ✅ 开发复杂的多Agent协作系统
- ✅ 实现工作流自动化和业务流程优化
- ✅ 进行模型微调和性能优化
- ✅ 设计和部署企业级AI Agent应用

---

## 📁 项目结构

```
Agent-101/
├── 00-agent-env/              # 环境搭建与配置
│   ├── linux_ops/            # Linux运维配置
│   ├── docker-ops/           # Docker环境配置
│   ├── jupyter-ops/          # Jupyter环境配置
│   └── setup_agent101_dev.sh   # 自动化配置脚本
├── 01-agent-prompt-or-context/  # 提示词工程与上下文管理
│   ├── prompt-enginner/      # 提示词工程实践
│   ├── context-engineer/     # 上下文管理
│   └── prompts_best_practice/ # 最佳实践案例
├── 02-agent-model-deploy/    # 模型部署与推理
├── 03-agent-mcp/             # MCP协议与工具集成
├── 04-agent-multi-role/      # 多角色Agent协作
│   ├── langchain/            # LangChain实现
│   ├── autogen/              # AutoGen实现
│   └── ango/                 # Ango框架实现
├── 05-agent-workflow-n8n/    # 工作流自动化
├── 06-agent-model-finetuning/ # 模型微调与优化
├── 07-agent-rag/             # RAG检索增强生成
│   ├── agentic_rag_math_agent/ # 数学Agent RAG
│   ├── qwen_local_rag/       # 本地Qwen RAG
│   └── rag_agent_cohere/     # Cohere RAG实现
├── 08-agent-project/         # 企业级项目实战
├── docs/                     # 项目文档
├── requirements.txt           # Python依赖
└── README.md                 # 项目说明
```

---

## 🚀 快速体验

### 第一个AI Agent应用

```bash
# 运行第一个LLM应用
python 00-agent-env/first_llm_app.py
```

### 预期输出

```
🚀 Agent-101: 第一个AI Agent应用
==================================================
✅ 客户端初始化成功
📡 API地址: https://api.openai.com/v1
🤖 使用模型: gpt-3.5-turbo

🚀 正在调用大模型...

🤖 AI助手回复:
📝 你好！我是一个AI助手，可以帮助你解答问题、提供信息、协助编程、翻译文本、创作内容等多种任务。

📊 使用统计:
🔤 输入tokens: 45
🔤 输出tokens: 28
🔤 总计tokens: 73

🎉 恭喜！您的第一个AI Agent应用运行成功！
```

---

## 📞 获取帮助

- 🐛 **Bug报告**: [GitHub Issues](https://github.com/FlyAIBox/Agent-101/issues)
- 💬 **技术讨论**: [GitHub Discussions](https://github.com/FlyAIBox/Agent-101/discussions)
- 📧 **邮件联系**: fly910905@sina.com
- 🔗 **微信公众号**: 萤火AI百宝箱

## 🙏 致谢

本项目使用了以下开源项目：

<table>
<tr>
<td align="center">
<img src="https://raw.githubusercontent.com/langchain-ai/.github/main/profile/logo-dark.svg#gh-light-mode-only" width="70">
<br>LangChain
</td>

<td align="center">
<img src="https://camo.githubusercontent.com/3fd5a9a03ec16da77b97a372a8cea9193dd6f1c30aba8f3f7222c1cf30c7e012/68747470733a2f2f61676e6f2d7075626c69632e73332e75732d656173742d312e616d617a6f6e6177732e636f6d2f6173736574732f6c6f676f2d6c696768742e737667" width="60">
<br>Ango
</td>

<td align="center">
<img src="https://raw.githubusercontent.com/microsoft/autogen/main/website/static/img/logo.png" width="60">
<br>AutoGen
</td>

<td align="center">
<img src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width="60">
<br>VLLM
</td>

<td align="center">
<img src="https://docs.unsloth.ai/~gitbook/image?url=https%3A%2F%2F2815821428-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Forganizations%252FHpyELzcNe0topgVLGCZY%252Fsites%252Fsite_mXXTe%252Flogo%252FccLeknrOqRa0v4q9P4Qh%252Funsloth%2520graffitti%2520black%2520text.png%3Falt%3Dmedia%26token%3D34deab0c-35f7-462c-8298-e7d8e2771a89&width=320&dpr=2&quality=100&sign=f8e8ce7a&sv=2" width="60">
<br>Unsloth
</td>
</tr>
</table>

特别感谢所有贡献者和社区成员的支持！

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给个Star支持！⭐**

<a href="https://star-history.com/#FlyAIBox/Agent-101&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=FlyAIBox/Agent-101&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=FlyAIBox/Agent-101&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=FlyAIBox/Agent-101&type=Date" />
  </picture>
</a>

**🔗 更多访问：[AI Agent实战101](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkzODUxMTY1Mg==&action=getalbum&album_id=3945699220593803270#wechat_redirect)**

</div>


