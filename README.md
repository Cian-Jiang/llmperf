# LLMPerf

A Tool for evaulation the performance of LLM APIs.

# Installation
```bash
git clone https://github.com/ray-project/llmperf.git
cd llmperf
pip install -e .
```

# Basic Usage

We implement 2 tests for evaluating LLMs: a load test to check for performance and a correctness test to check for correctness.

## Load test

The load test spawns a number of concurrent requests to the LLM API and measures the inter-token latency and generation throughput per request and across concurrent requests. The prompt that is sent with each request is of the format:

```
Randomly stream lines from the following text. Don't generate eos tokens:
LINE 1,
LINE 2,
LINE 3,
...
```

Where the lines are randomly sampled from a collection of lines from Shakespeare sonnets. Tokens are counted using the `LlamaTokenizer` regardless of which LLM API is being tested. This is to ensure that the prompts are consistent across different LLM APIs.

To run the most basic load test you can the token_benchmark_ray script.


### Caveats and Disclaimers

- The endpoints provider backend might vary widely, so this is not a reflection on how the software runs on a particular hardware.
- The results may vary with time of day.
- The results may vary with the load.
- The results may not correlate with users’ workloads.

### OpenAI Compatible APIs
```bash
export OPENAI_API_KEY="ollama"
export OPENAI_API_BASE="http://localhost:11434/v1"

python3 token_benchmark_ray.py \
--model "hf.co/rombodawg/Rombos-LLM-V2.6-Qwen-14b-Q8_0-GGUF:latest" \
--mean-input-tokens 230 \
--stddev-input-tokens 100 \
--mean-output-tokens 280 \
--stddev-output-tokens 100 \
--max-num-completed-requests 16 \
--timeout 600 \
--num-concurrent-requests 4 \
--results-dir "result_outputs" \
--llm-api openai \
--additional-sampling-params '{}'

```
#### result
```bash
--mean-input-tokens 230 \ # 平均输入长度
--stddev-input-tokens 100 \ # 输入长度标准差
--mean-output-tokens 280 \ # 目标输出长度
--stddev-output-tokens 100 \ # 输出长度标准差
--max-num-completed-requests 16 \ # 总请求数
--timeout 600 \ 
--num-concurrent-requests 4 \ # 并发请求数
--results-dir "result_outputs" \
--llm-api openai \
--additional-sampling-params '{}'
```
# 本地ollama大模型全尺寸性能对比分析报告

## 测试配置概览

| 配置项 | Gemma2 27B | Qwen2.5 32B | Qwen2.5 14B | Gemma2 9B | Qwen2.5 7B | Llama3.1 8B |
|--------|------------|-------------|-------------|------------|------------|-------------|
| 模型名称 | gemma2:27b-instruct-q5_K_M | qwen2.5:32b-instruct-q4_K_M | qwen2.5:14b-instruct-q5_K_M | gemma2:9b-instruct-q8_0 | qwen2.5:7b-instruct-q8_0 | llama3.1:8b-instruct-q8_0 |
| 量化精度 | Q5_K_M | Q4_K_M | Q5_K_M | Q8_0 | Q8_0 | Q8_0 |
| 参数规模 | 27B | 32B | 14B | 9B | 7B | 8B |

## 性能指标详细对比

### 1. 响应速度指标

#### 首个令牌生成时间 (TTFT, 单位: 秒)
| 指标 | Gemma2 27B | Qwen2.5 32B | Qwen2.5 14B | Gemma2 9B | Qwen2.5 7B | Llama3.1 8B |
|------|------------|-------------|-------------|------------|------------|-------------|
| 平均值 | 9.36 | 9.91 | 3.80 | 3.44 | 2.21 | 2.43 |
| 中位数 | 9.35 | 10.39 | 4.49 | 2.78 | 1.81 | 1.99 |
| 最快响应 | 1.11 | 1.21 | 0.76 | 0.44 | 0.39 | 0.34 |
| P95 | 18.80 | 19.12 | 5.51 | 8.25 | 5.36 | 5.91 |
| 标准差 | 5.58 | 5.57 | 1.63 | 2.69 | 1.75 | 1.91 |

#### 令牌间延迟 (单位: 秒)
| 指标 | Gemma2 27B | Qwen2.5 32B | Qwen2.5 14B | Gemma2 9B | Qwen2.5 7B | Llama3.1 8B |
|------|------------|-------------|-------------|------------|------------|-------------|
| 平均值 | 0.409 | 0.312 | 0.186 | 0.095 | 0.062 | 0.066 |
| 中位数 | 0.419 | 0.318 | 0.185 | 0.089 | 0.064 | 0.064 |
| 标准差 | 0.058 | 0.047 | 0.025 | 0.021 | 0.011 | 0.013 |

#### 端到端延迟 (单位: 秒)
| 指标 | Gemma2 27B | Qwen2.5 32B | Qwen2.5 14B | Gemma2 9B | Qwen2.5 7B | Llama3.1 8B |
|------|------------|-------------|-------------|------------|------------|-------------|
| 平均值 | 66.36 | 57.58 | 46.95 | 11.77 | 10.67 | 10.49 |
| 中位数 | 70.03 | 58.59 | 44.88 | 12.06 | 10.20 | 10.48 |
| P95 | 92.92 | 87.33 | 62.66 | 16.54 | 16.04 | 15.46 |
| 标准差 | 19.23 | 17.62 | 11.34 | 3.60 | 3.39 | 3.16 |

### 2. 吞吐量指标

#### 单请求吞吐量 (tokens/秒)
| 指标 | Gemma2 27B | Qwen2.5 32B | Qwen2.5 14B | Gemma2 9B | Qwen2.5 7B | Llama3.1 8B |
|------|------------|-------------|-------------|------------|------------|-------------|
| 平均值 | 2.49 | 3.28 | 5.47 | 10.85 | 16.46 | 15.45 |
| 中位数 | 2.37 | 3.12 | 5.40 | 11.14 | 15.52 | 15.45 |
| 最高值 | 3.47 | 5.15 | 7.57 | 14.52 | 21.42 | 20.76 |
| 标准差 | 0.42 | 0.62 | 0.80 | 2.17 | 2.90 | 2.71 |

#### 整体性能
| 指标 | Gemma2 27B | Qwen2.5 32B | Qwen2.5 14B | Gemma2 9B | Qwen2.5 7B | Llama3.1 8B |
|------|------------|-------------|-------------|------------|------------|-------------|
| 总体吞吐量 (tok/s) | 8.90 | 11.49 | 19.47 | 35.34 | 48.41 | 50.77 |
| 请求处理速率 (req/min) | 3.19 | 3.62 | 4.51 | 16.99 | 16.58 | 19.09 |

### 3. 输出特征
| 指标 | Gemma2 27B | Qwen2.5 32B | Qwen2.5 14B | Gemma2 9B | Qwen2.5 7B | Llama3.1 8B |
|------|------------|-------------|-------------|------------|------------|-------------|
| 平均输出令牌数 | 167.38 | 190.69 | 259.0 | 124.81 | 175.19 | 159.56 |
| 标准差 | 67.94 | 73.33 | 78.85 | 38.79 | 64.42 | 49.79 |
| 最大输出令牌数 | 350 | 352 | 409 | 203 | 323 | 279 |

## 性能分析

### 1. 模型尺寸与性能关系分析

#### Gemma2 系列规模效应（9B vs 27B）
1. **响应速度变化**
    - TTFT: 增加 172% (3.44s → 9.36s)
    - 令牌间延迟: 增加 330% (0.095s → 0.409s)
    - 端到端延迟: 增加 464% (11.77s → 66.36s)

2. **吞吐量变化**
    - 单请求吞吐量: 下降 77% (10.85 → 2.49)
    - 总体吞吐量: 下降 75% (35.34 → 8.90)

#### Qwen2.5 系列规模效应（7B → 14B → 32B）
1. **响应速度变化**
    - TTFT: 7B → 32B 增加 348% (2.21s → 9.91s)
    - 令牌间延迟: 7B → 32B 增加 403% (0.062s → 0.312s)
    - 端到端延迟: 7B → 32B 增加 440% (10.67s → 57.58s)

2. **吞吐量变化**
    - 单请求吞吐量: 7B → 32B 下降 80% (16.46 → 3.28)
    - 总体吞吐量: 7B → 32B 下降 76% (48.41 → 11.49)

### 2. 规模梯度效应

1. **小规模模型（7B-9B）**
    - 最佳响应速度
    - 最高吞吐量
    - 资源效率最优

2. **中规模模型（14B）**
    - 平衡的性能表现
    - 较好的输出长度
    - 适中的资源消耗

3. **大规模模型（27B-32B）**
    - 最长响应时间
    - 最低吞吐量
    - 最高资源需求

### 3. 量化效果分析

1. **量化精度影响**
    - Q8_0: 最佳性能表现（小模型）
    - Q5_K_M: 中等性能损失
    - Q4_K_M: 显著性能损失

2. **规模与量化关系**
    - 较大模型对量化更敏感
    - 量化精度对吞吐量影响显著

## 部署建议

### 1. 场景匹配

#### 实时交互场景
- **推荐**: Llama3.1 8B, Qwen2.5 7B
- **优势**:
    - 最快的响应速度
    - 最高的吞吐量
    - 资源消耗最优

#### 高质量输出场景
- **推荐**: Qwen2.5 32B, Gemma2 27B
- **优势**:
    - 更强的推理能力
    - 更丰富的知识储备
    - 更好的生成质量

#### 平衡场景
- **推荐**: Qwen2.5 14B, Gemma2 9B
- **优势**:
    - 平衡的性能表现
    - 适中的资源消耗
    - 较好的输出质量

### 2. 部署策略

#### 单一模型部署
1. **资源受限环境**
    - 首选 Llama3.1 8B 或 Qwen2.5 7B
    - 考虑使用 Q8_0 量化

2. **资源充足环境**
    - 可选择 Qwen2.5 32B 或 Gemma2 27B
    - 权衡量化精度和性能

#### 混合部署策略
1. **基于任务类型**
    - 简单任务：小模型处理
    - 复杂任务：大模型处理

2. **基于负载**
    - 高峰期：增加小模型实例
    - 低峰期：保持大模型服务

## 优化建议

### 1. 系统层面
1. **资源管理**
    - 优化内存使用
    - 实现动态负载均衡
    - 合理配置缓存策略

2. **并发控制**
    - 根据模型规模调整并发度
    - 实现请求排队机制
    - 设置超时控制

### Anthropic
```bash
export ANTHROPIC_API_KEY=secret_abcdefg

python token_benchmark_ray.py \
--model "claude-2" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api anthropic \
--additional-sampling-params '{}'

```

### TogetherAI

```bash
export TOGETHERAI_API_KEY="YOUR_TOGETHER_KEY"

python token_benchmark_ray.py \
--model "together_ai/togethercomputer/CodeLlama-7b-Instruct" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api "litellm" \
--additional-sampling-params '{}'

```

### Hugging Face

```bash
export HUGGINGFACE_API_KEY="YOUR_HUGGINGFACE_API_KEY"
export HUGGINGFACE_API_BASE="YOUR_HUGGINGFACE_API_ENDPOINT"

python token_benchmark_ray.py \
--model "huggingface/meta-llama/Llama-2-7b-chat-hf" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api "litellm" \
--additional-sampling-params '{}'

```

### LiteLLM

LLMPerf can use LiteLLM to send prompts to LLM APIs. To see the environment variables to set for the provider and arguments that one should set for model and additional-sampling-params.

see the [LiteLLM Provider Documentation](https://docs.litellm.ai/docs/providers).

```bash
python token_benchmark_ray.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api "litellm" \
--additional-sampling-params '{}'

```

### Vertex AI

Here, --model is used for logging, not for selecting the model. The model is specified in the Vertex AI Endpoint ID.

The GCLOUD_ACCESS_TOKEN needs to be somewhat regularly set, as the token generated by `gcloud auth print-access-token` expires after 15 minutes or so.

Vertex AI doesn't return the total number of tokens that are generated by their endpoint, so tokens are counted using the LLama tokenizer.

```bash

gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

export GCLOUD_ACCESS_TOKEN=$(gcloud auth print-access-token)
export GCLOUD_PROJECT_ID=YOUR_PROJECT_ID
export GCLOUD_REGION=YOUR_REGION
export VERTEXAI_ENDPOINT_ID=YOUR_ENDPOINT_ID

python token_benchmark_ray.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api "vertexai" \
--additional-sampling-params '{}'

```

### SageMaker

SageMaker doesn't return the total number of tokens that are generated by their endpoint, so tokens are counted using the LLama tokenizer.

```bash

export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_ACCESS_KEY"s
export AWS_SESSION_TOKEN="YOUR_SESSION_TOKEN"
export AWS_REGION_NAME="YOUR_ENDPOINTS_REGION_NAME"

python llm_correctness.py \
--model "llama-2-7b" \
--llm-api "sagemaker" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

see `python token_benchmark_ray.py --help` for more details on the arguments.

## Correctness Test

The correctness test spawns a number of concurrent requests to the LLM API with the following format:

```
Convert the following sequence of words into a number: {random_number_in_word_format}. Output just your final answer.
```

where random_number_in_word_format could be for example "one hundred and twenty three". The test then checks that the response contains that number in digit format which in this case would be 123.

The test does this for a number of randomly generated numbers and reports the number of responses that contain a mismatch.

To run the most basic correctness test you can run the the llm_correctness.py script.

### OpenAI Compatible APIs

```bash
export OPENAI_API_KEY=secret_abcdefg
export OPENAI_API_BASE=https://console.endpoints.anyscale.com/m/v1

python llm_correctness.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--max-num-completed-requests 150 \
--timeout 600 \
--num-concurrent-requests 10 \
--results-dir "result_outputs"
```

### Anthropic

```bash
export ANTHROPIC_API_KEY=secret_abcdefg

python llm_correctness.py \
--model "claude-2" \
--llm-api "anthropic"  \
--max-num-completed-requests 5 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs"
```

### TogetherAI

```bash
export TOGETHERAI_API_KEY="YOUR_TOGETHER_KEY"

python llm_correctness.py \
--model "together_ai/togethercomputer/CodeLlama-7b-Instruct" \
--llm-api "litellm" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

### Hugging Face

```bash
export HUGGINGFACE_API_KEY="YOUR_HUGGINGFACE_API_KEY"
export HUGGINGFACE_API_BASE="YOUR_HUGGINGFACE_API_ENDPOINT"

python llm_correctness.py \
--model "huggingface/meta-llama/Llama-2-7b-chat-hf" \
--llm-api "litellm" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

### LiteLLM

LLMPerf can use LiteLLM to send prompts to LLM APIs. To see the environment variables to set for the provider and arguments that one should set for model and additional-sampling-params.

see the [LiteLLM Provider Documentation](https://docs.litellm.ai/docs/providers).

```bash
python llm_correctness.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--llm-api "litellm" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

see `python llm_correctness.py --help` for more details on the arguments.


### Vertex AI

Here, --model is used for logging, not for selecting the model. The model is specified in the Vertex AI Endpoint ID.

The GCLOUD_ACCESS_TOKEN needs to be somewhat regularly set, as the token generated by `gcloud auth print-access-token` expires after 15 minutes or so.

Vertex AI doesn't return the total number of tokens that are generated by their endpoint, so tokens are counted using the LLama tokenizer.


```bash

gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

export GCLOUD_ACCESS_TOKEN=$(gcloud auth print-access-token)
export GCLOUD_PROJECT_ID=YOUR_PROJECT_ID
export GCLOUD_REGION=YOUR_REGION
export VERTEXAI_ENDPOINT_ID=YOUR_ENDPOINT_ID

python llm_correctness.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--llm-api "vertexai" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

### SageMaker

SageMaker doesn't return the total number of tokens that are generated by their endpoint, so tokens are counted using the LLama tokenizer.

```bash

export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_ACCESS_KEY"s
export AWS_SESSION_TOKEN="YOUR_SESSION_TOKEN"
export AWS_REGION_NAME="YOUR_ENDPOINTS_REGION_NAME"

python llm_correctness.py \
--model "llama-2-7b" \
--llm-api "sagemaker" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

## Saving Results

The results of the load test and correctness test are saved in the results directory specified by the `--results-dir` argument. The results are saved in 2 files, one with the summary metrics of the test, and one with metrics from each individual request that is returned.

# Advanced Usage

The correctness tests were implemented with the following workflow in mind:

```python
import ray
from transformers import LlamaTokenizerFast

from llmperf.ray_clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
)
from llmperf.models import RequestConfig
from llmperf.requests_launcher import RequestsLauncher


# Copying the environment variables and passing them to ray.init() is necessary
# For making any clients work.
ray.init(runtime_env={"env_vars": {"OPENAI_API_BASE" : "https://api.endpoints.anyscale.com/v1",
                                   "OPENAI_API_KEY" : "YOUR_API_KEY"}})

base_prompt = "hello_world"
tokenizer = LlamaTokenizerFast.from_pretrained(
    "hf-internal-testing/llama-tokenizer"
)
base_prompt_len = len(tokenizer.encode(base_prompt))
prompt = (base_prompt, base_prompt_len)

# Create a client for spawning requests
clients = [OpenAIChatCompletionsClient.remote()]

req_launcher = RequestsLauncher(clients)

req_config = RequestConfig(
    model="meta-llama/Llama-2-7b-chat-hf",
    prompt=prompt
    )

req_launcher.launch_requests(req_config)
result = req_launcher.get_next_ready(block=True)
print(result)

```

# Implementing New LLM Clients

To implement a new LLM client, you need to implement the base class `llmperf.ray_llm_client.LLMClient` and decorate it as a ray actor.

```python

from llmperf.ray_llm_client import LLMClient
import ray


@ray.remote
class CustomLLMClient(LLMClient):

    def llm_request(self, request_config: RequestConfig) -> Tuple[Metrics, str, RequestConfig]:
        """Make a single completion request to a LLM API

        Returns:
            Metrics about the performance charateristics of the request.
            The text generated by the request to the LLM API.
            The request_config used to make the request. This is mainly for logging purposes.

        """
        ...

```

# Legacy Codebase
The old LLMPerf code base can be found in the [llmperf-legacy](https://github.com/ray-project/llmval-legacy) repo.
