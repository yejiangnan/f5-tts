#!/usr/bin/env python3
# coding: utf-8

import os
import json
import glob
from tqdm import tqdm
from openai import OpenAI
import threading
import queue
import fcntl
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

## 模型文档
## https://wvixbzgc0u7.feishu.cn/wiki/YJWJw5LY9iJnwrkITchcUcpynoc


PROMPT="""你是一位专业文本作家，你需要给我50句中文：
1. 每句话单独占据一行，但是不要序号。每句话的字数在50字左右。
2. 保证每句话包含一个英文缩写，这个英文缩写是由大写字母构成，同时英文母语者读这个缩写的方式是逐个字母读出。同时缩写要有多样性，不要重复。
3. 英文缩写出现的位子要有多样性：在文本的开头、中前、中后、结尾。
4. 不要任何输出解释、说明。
"""

PROMPT_EMPHASIS="""你是一位专业的语音语调分析专家。你的任务是识别文本中**最有可能被重读的词**。

**核心原则：绝大多数文本不需要重读词，只有在非常明确的强调、对比、疑问、否定语境中才可能有重读词。宁可输出"无"，也不要过度识别。**

**严格的重读词选择标准（必须同时满足以下所有条件）**：
1. **明确的疑问、否定或对比语境**：只有在疑问句、否定句、对比句中才可能重读
2. **特定的功能词**：只有疑问词（什么、哪里、怎么、为什么、哪个）、否定词（不、没、非、无、别）、对比词（但是、然而、反而、却、而是）、强调词（非常、特别、确实、真的、一定）才可能重读
3. **强烈的强调意图**：只有在表达强烈情感、态度、观点或进行明确对比时才重读
4. **自然说话习惯**：必须符合人们日常说话时的自然重读习惯，不能是生硬或不符合语言习惯的选择

**以下情况绝对不需要重读（绝大多数情况属于此类）**：
- 普通陈述句中的任何词（即使是有意义的词）
- 普通的名词、动词、形容词（除非在明确的对比语境中）
- 所有连接词、助词、语气词（的、了、在、和、也、都等）
- 普通的时间、地点、数字词（除非是明确的对比焦点）
- 描述性、叙述性的词
- 没有明确强调、对比、疑问、否定意图的任何词

**只有在以下极其明确的情况下才可能重读**：
- 疑问句中的疑问词（什么、哪里、怎么、为什么、哪个）
- 否定句中的否定词（不、没、非、无、别），且该否定词是句子的焦点
- 对比句中的对比词（但是、然而、反而、却、而是）或对比焦点词
- 表达强烈态度时的强调词（非常、特别、确实、真的、一定），且该词是句子的核心强调点

**重要：即使是疑问词、否定词、对比词，如果它们不是句子的焦点，也不需要重读。**

输出要求：
1. **绝大多数情况下应该输出"无"**：只有在非常明确的强调、对比、疑问、否定语境中才输出重读词
2. **只输出最有可能被重读的词，每个词单独占一行（每行一个词）**
3. **如果文本中没有明显需要重读的词，只输出"无"，不要输出其他内容**
4. 必须从原文中精确提取，不能改写、替换或添加任何内容
5. 只输出词本身，不要加任何标记、解释或说明
6. 不要输出标点符号
7. 如果文本为空或只有标点，输出"无"
8. 每个词必须是独立的词，不能是短语
9. **严格标准：宁可输出"无"，也不要多识别任何一个词**

示例：
输入：他今天为什么没有来参加会议？
输出：
为什么
没有

输入：这个项目的预算确实超出了预期。
输出：
确实

输入：我昨天去了北京，不是上海。
输出：
不是

输入：今天天气很好，我们去公园散步吧。
输出：
无

输入：他今天去了公司，处理了一些工作。
输出：
无

输入：这个方案非常好，我们都很满意。
输出：
无

现在请分析以下文本，**严格按照标准，只找出最有可能被重读的词**（绝大多数情况下应该输出"无"）：

{input_text}
"""

PROMPT_EMPHASIS_BATCH="""你是一位专业的语音语调分析专家。你的任务是识别文本中**最有可能被重读的词**。

**核心原则：绝大多数文本不需要重读词，只有在非常明确的强调、对比、疑问、否定语境中才可能有重读词。宁可输出"无"，也不要过度识别。**

**严格的重读词选择标准（必须同时满足以下所有条件）**：
1. **明确的疑问、否定或对比语境**：只有在疑问句、否定句、对比句中才可能重读
2. **特定的功能词**：只有疑问词（什么、哪里、怎么、为什么、哪个）、否定词（不、没、非、无、别）、对比词（但是、然而、反而、却、而是）、强调词（非常、特别、确实、真的、一定）才可能重读
3. **强烈的强调意图**：只有在表达强烈情感、态度、观点或进行明确对比时才重读
4. **自然说话习惯**：必须符合人们日常说话时的自然重读习惯，不能是生硬或不符合语言习惯的选择

**以下情况绝对不需要重读（绝大多数情况属于此类）**：
- 普通陈述句中的任何词（即使是有意义的词）
- 普通的名词、动词、形容词（除非在明确的对比语境中）
- 所有连接词、助词、语气词（的、了、在、和、也、都等）
- 普通的时间、地点、数字词（除非是明确的对比焦点）
- 描述性、叙述性的词
- 没有明确强调、对比、疑问、否定意图的任何词

**只有在以下极其明确的情况下才可能重读**：
- 疑问句中的疑问词（什么、哪里、怎么、为什么、哪个）
- 否定句中的否定词（不、没、非、无、别），且该否定词是句子的焦点
- 对比句中的对比词（但是、然而、反而、却、而是）或对比焦点词
- 表达强烈态度时的强调词（非常、特别、确实、真的、一定），且该词是句子的核心强调点

**重要：即使是疑问词、否定词、对比词，如果它们不是句子的焦点，也不需要重读。**

输出要求：
1. **绝大多数情况下应该输出"无"**：只有在非常明确的强调、对比、疑问、否定语境中才输出重读词
2. **只输出最有可能被重读的词，每个词单独占一行（每行一个词）**
3. **如果文本中没有明显需要重读的词，只输出"无"，不要输出其他内容**
4. 必须从原文中精确提取，不能改写、替换或添加任何内容
5. 只输出词本身，不要加任何标记、解释或说明
6. 不要输出标点符号
7. 如果文本为空或只有标点，输出"无"
8. 每个词必须是独立的词，不能是短语
9. **严格标准：宁可输出"无"，也不要多识别任何一个词**

**重要：以下是多条文本，请为每条文本分别分析并输出结果。**
**输出格式：每条文本的结果用"===文本N==="作为分隔符，N是文本的序号（从1开始）。每条文本的结果格式与单条文本相同。**

示例：
输入：
===文本1===
他今天为什么没有来参加会议？
===文本2===
这个项目的预算确实超出了预期。
===文本3===
我昨天去了北京，不是上海。
===文本4===
今天天气很好，我们去公园散步吧。
===文本5===
他今天去了公司，处理了一些工作。
===文本6===
这个方案非常好，我们都很满意。

输出：
===文本1===
为什么
没有
===文本2===
确实
===文本3===
不是
===文本4===
无
===文本5===
无
===文本6===
无

现在请分析以下多条文本，**严格按照标准，只找出每条文本中最有可能被重读的词**（绝大多数情况下应该输出"无"）：

{input_text}
"""

PROMPT_NUMBER="""你是一个专业的、不会犯错的文本修改大师，请帮我做文本改写。要求如下：
- 只改写文本中的数字，不要动其余部分
- 我需要一些重复数字，比如（二二二三零的二，零点一七七七七的七）
   - 如果文本中本来就有重复的数字，你可以选择增加一个，或者把那个数字替换成别的，但要注意，修改后的数字必须是有意义的（比如一点零零三改成一点零零零三是有意义的，但是改成一一点零零三是错的）
  - 如果文本中本来没有重复的数字，请增加一个符合规则的重复数字
- 对于二零二五年这样的年份，请修改成别的年份
- 对于数字+亿这样的组合，请修改成一一亿或者一一一亿

用户提供的文本是：
{input_text}
"""


def request_LLM_for_expansion(model_name, client, input_text, use_gpt4o=False, max_retries=8, initial_delay=2, prompt_type="emphasis", is_batch=False):
    """调用LLM API获取扩展后的响应
    
    Args:
        input_text: 单条文本（字符串）或批量文本（字符串列表）
        is_batch: 是否为批量处理模式
    """
    if is_batch and isinstance(input_text, list):
        # 批量处理模式：构建批量 prompt
        if prompt_type == "emphasis":
            batch_input = "\n".join([f"===文本{i+1}===\n{text}" for i, text in enumerate(input_text)])
            prompt = PROMPT_EMPHASIS_BATCH.format(input_text=batch_input)
        else:
            # 其他类型暂不支持批量
            raise ValueError(f"批量处理暂不支持 prompt_type={prompt_type}")
    else:
        # 单条处理模式
        if prompt_type == "emphasis":
            prompt = PROMPT_EMPHASIS.format(input_text=input_text)
        elif prompt_type == "number":
            prompt = PROMPT_NUMBER.format(input_text=input_text)
        else:
            prompt = PROMPT_NUMBER.format(input_text=input_text)
    
    # print(f"[prompt] {prompt}]")
    model = model_name
    for attempt in range(max_retries):
        try:
            if use_gpt4o:
                # GPT-4o 调用方式（纯文本版本）
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
            else:
                # DeepSeek 调用方式
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                model_type = "GPT-4o" if use_gpt4o else "DeepSeek"
                print(f"{model_type} expansion failed after {max_retries} retries. Error: {e}")
                return None
            delay = min(initial_delay * (2 ** attempt), 60)
            model_type = "GPT-4o" if use_gpt4o else "DeepSeek"
            print(f"{model_type} API调用失败 (扩展任务)，重试{attempt+1}：{e}，等待{delay:.1f}s")
            time.sleep(delay)
    return None

def process_request(args, line:str):
    if line.startswith("{"):
        try:
            sample = json.loads(line.strip())
            input_text = sample['vad_segments'][0]['text']
        except Exception as e:
            print(f"[Error] {line} {e}")
            return False, {
                "input_text": line.strip(),
                "error": str(e)
            }
    else:
        input_text = line.strip()

    request_result = request_LLM_for_expansion(
        model_name=args.model_name, 
        client=args.client, 
        input_text=input_text, 
        use_gpt4o=args.use_gpt4o,
        prompt_type=args.prompt_type
    )
    
    if request_result is not None:
        # 如果 prompt_type 是 emphasis，将多个词拆分成多个结果项
        if args.prompt_type == "emphasis":
            # 按行分割，过滤空行和"无"
            emphasis_words = [
                word.strip() 
                for word in request_result.split('\n') 
                if word.strip() and word.strip() != "无"
            ]
            
            # 如果没有找到重读词，跳过（返回特殊标记）
            if not emphasis_words:
                return "skip", None  # 跳过，不输出任何结果
            
            # 返回多个结果项的列表
            return True, [
                {
                    "input_text": input_text,
                    "text": word,
                }
                for word in emphasis_words
            ]
        else:
            # 其他 prompt_type 保持原样
            sample = {
                "input_text": input_text,
                "text": request_result,
            }
            return True, sample
    return False, {
        "input_text": input_text,
        "error": "API调用失败"
    }

def parse_batch_result(batch_result, num_texts):
    """解析批量处理的结果
    
    Returns:
        list: 每个元素是重读词列表（可能为空表示跳过）
    """
    results = []
    if not batch_result:
        return [[] for _ in range(num_texts)]
    
    # 按 "===文本N===" 分割结果
    parts = batch_result.split("===文本")
    for i in range(1, len(parts)):  # 跳过第一个空部分
        part = parts[i]
        # 提取文本序号和内容
        if "===" in part:
            content = part.split("===", 1)[1].strip()
            # 提取重读词（按行分割，过滤空行和"无"）
            emphasis_words = [
                word.strip() 
                for word in content.split('\n') 
                if word.strip() and word.strip() != "无"
            ]
            results.append(emphasis_words)
        else:
            # 格式异常，添加空列表
            results.append([])
    
    # 确保结果数量匹配
    while len(results) < num_texts:
        results.append([])
    
    return results[:num_texts]

def process_batch_request(args, lines: list):
    """批量处理多条文本
    
    Args:
        args: 参数对象（包含 batch_size）
        lines: 文本行列表（已经是一个批次）
    
    Returns:
        list: 每个元素是 (success, result) 的元组，与 process_request 的返回格式一致
    """
    results = []
    batch_size = args.batch_size
    
    # 将文本分批处理（如果 lines 超过 batch_size，需要再次分批）
    for i in range(0, len(lines), batch_size):
        batch_lines = lines[i:i+batch_size]
        batch_texts = []
        batch_original_lines = []
        
        # 提取每条文本的 input_text
        for line in batch_lines:
            if line.startswith("{"):
                try:
                    sample = json.loads(line.strip())
                    input_text = sample['vad_segments'][0]['text']
                except Exception as e:
                    # 单条处理失败的情况
                    results.append((False, {
                        "input_text": line.strip(),
                        "error": str(e)
                    }))
                    continue
            else:
                input_text = line.strip()
            
            batch_texts.append(input_text)
            batch_original_lines.append((line, input_text))
        
        if not batch_texts:
            continue
        
        # 批量调用 API
        request_result = request_LLM_for_expansion(
            model_name=args.model_name, 
            client=args.client, 
            input_text=batch_texts, 
            use_gpt4o=args.use_gpt4o,
            prompt_type=args.prompt_type,
            is_batch=True
        )
        
        if request_result is not None:
            # 解析批量结果
            batch_results = parse_batch_result(request_result, len(batch_texts))
            
            # 为每条文本生成结果
            for (line, input_text), emphasis_words in zip(batch_original_lines, batch_results):
                if args.prompt_type == "emphasis":
                    if not emphasis_words:
                        # 跳过没有重读词的文本
                        results.append(("skip", None))
                    else:
                        # 返回多个结果项的列表
                        results.append((True, [
                            {
                                "input_text": input_text,
                                "text": word,
                            }
                            for word in emphasis_words
                        ]))
                else:
                    # 其他类型暂不支持批量
                    results.append((False, {
                        "input_text": input_text,
                        "error": "批量处理暂不支持此 prompt_type"
                    }))
        else:
            # API 调用失败，为每条文本返回错误
            for _, input_text in batch_original_lines:
                results.append((False, {
                    "input_text": input_text,
                    "error": "LLM API调用失败"
                }))
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch process audio JSONs with reasoning LLM.")
    parser.add_argument("--input_file", required=True, help="Directory containing .tar files")
    # parser.add_argument("--input_emotion", required=False, help="Emotion of the target text.")
    parser.add_argument("--save_file_path", required=True, help="Directory to write processed JSONs")
    parser.add_argument("--error_file_path", required=True, help="Directory to write processed JSONs of error")
    parser.add_argument("--max_workers", type=int, default=1, help="Number of concurrent threads")
    parser.add_argument("--world_size", type=int, default=8, help="Number of concurrent threads")
    parser.add_argument("--rank", type=int, default=8, help="Number of concurrent threads")
    parser.add_argument("--use_gpt4o", type=str, default="false", help="model server")
    parser.add_argument("--prompt_type", type=str, default="emphasis", help="prompt type: emphasis, number, etc.")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of texts to process in one API call (batch mode)")
    args = parser.parse_args()

    if args.use_gpt4o == "true":
        args.use_gpt4o = True
    else:
        args.use_gpt4o = False

    # LLM API 设置
    if args.use_gpt4o:
        args.model_name = "gpt-4o"  # 纯文本版本的GPT-4o
        # GPT-4o 配置 (参考 gpt4o_audio_preview_multitask.py)
        gpt4o_client = OpenAI(
            api_key="ak-x4h863dfvomqif7nj4sit2gzfrvrfkxm",
            base_url="https://models-proxy.stepfun-inc.com/v1",
            timeout=600
        )
        args.client = gpt4o_client
    else:
        args.model_name = "deepseek-r1-volce"
        # DeepSeek 配置
        deepseek_client = OpenAI(
            # api_key="ak-63h2ijkl89m0nop12qrs34tuvw56xyz7a8",
            # api_key="ak-57d1efgh23i9jkl64mno32pqrs18tuv4k6",
            api_key="ak-x4h863dfvomqif7nj4sit2gzfrvrfkxm",
            base_url="https://models-proxy.stepfun-inc.com/v1",
            timeout=600
        )
        args.client = deepseek_client

    write_lock = threading.Lock()
    error_lock = threading.Lock()

    world_size = args.world_size
    rank = args.rank

    done_set = set()
    if os.path.exists(args.save_file_path):
        with open(args.save_file_path, 'r', encoding='utf-8')as f:
            done_lines = f.readlines()

        ## 已经跑完的，先存储（记录所有已处理的 input_text）
        for done_line in tqdm(done_lines, desc="加载已完成记录"):
            try:
                data = json.loads(done_line.strip())
                text = data.get('input_text', '')
                if text:
                    done_set.add(text)
            except:
                pass

    # 优化大文件处理：流式读取，不一次性加载所有行到内存
    undo_lines = []
    len_done_set = len(done_set)
    
    # 先统计总行数（用于进度条）
    total_lines = 0
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    
    # 流式读取并过滤已完成的行
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="读取文件"):
            line_stripped = line.strip()
            if not line_stripped:  # 跳过空行
                continue
            if len_done_set == 0:
                undo_lines.append(line_stripped)
            else:
                # 检查是否已完成（通过 input_text 匹配）
                if line_stripped not in done_set:
                    undo_lines.append(line_stripped)
    
    print(f"[Info] 总行数: {total_lines}, 已完成: {len_done_set}, 待完成: {len(undo_lines)}")
    print(f"[Info] 使用批量处理模式，每批处理 {args.batch_size} 条文本，并发线程数: {args.max_workers}")
    
    # 将待处理的行分批
    batch_groups = []
    for i in range(0, len(undo_lines), args.batch_size):
        batch_groups.append(undo_lines[i:i+args.batch_size])
    
    # 使用线程池并发处理各个批次
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_batch = {executor.submit(process_batch_request, args, batch): batch for batch in batch_groups}
        for future in tqdm(as_completed(future_to_batch), total=len(batch_groups), desc="处理批次"):
            try:
                batch_results = future.result()
                # 处理批量返回的结果
                for succ, result in batch_results:
                    if succ == "skip":
                        # 跳过没有明显重读词的文本，不输出任何内容
                        continue
                    elif succ:
                        with write_lock:
                            with open(args.save_file_path, 'a', encoding='utf-8')as f:
                                # 如果 result 是列表（多个重读词），每个词写一行
                                if isinstance(result, list):
                                    for item in result:
                                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                                else:
                                    # 单个结果，直接写入
                                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    else:
                        with error_lock:
                            with open(args.error_file_path, 'a', encoding='utf-8')as f:
                                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            except Exception as e:
                time.sleep(1)
                print(f"Generated an exception: {e}")

def test():
    model_name = "deepseek-r1-volce"
    # DeepSeek 配置
    deepseek_client = OpenAI(
        api_key="ak-3a4b5c6d7e8f9g0h1i2j3k4l5m6n7o8p9",
        base_url="https://models-proxy.stepfun-inc.com/v1",
        timeout=600
    )
    client = deepseek_client
    res = request_LLM_for_expansion(model_name, client, "他们或许，在当时都坐着，呃，同类型的火车，住在沿海的类似的城中村，去类似的，呃，厂里先打工。但是或许就是因为某个机遇。", use_gpt4o=False)
    print(res)

if __name__ == "__main__":
    main()
    # test()
