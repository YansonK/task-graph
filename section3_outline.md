# Section 3: Background & Related Work - Detailed Outline

## 3.1 Language Model Optimization Approaches (1-1.5 pages)

### 3.1.1 Prompting Strategies

**Zero-shot and Few-shot Learning**
- Explain how few-shot examples guide LLM behavior
- Discuss limitations: brittleness, context window constraints
- *No specific citation needed - general background*

**Chain-of-Thought (CoT) Prompting**
- Core idea: explicit reasoning traces improve complex reasoning
- Key results: ~100B parameter threshold for emergence
- Performance gains on arithmetic, commonsense, symbolic reasoning
- **Cite:** Wei et al. (2022) "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" [[arxiv]](https://arxiv.org/abs/2201.11903)
  - 540B model with 8 CoT exemplars achieves SOTA on GSM8K math benchmark
  - **Critical point for your paper:** CoT helps but still requires manual prompt engineering

**ReAct: Reasoning + Acting**
- Interleaving reasoning traces with tool actions
- Overcomes hallucination by grounding in external knowledge (Wikipedia API)
- Outperforms pure reasoning on HotpotQA, Fever, ALFWorld, WebShop
- **Cite:** Yao et al. (2023) "ReAct: Synergizing Reasoning and Acting in Language Models" [[arxiv]](https://arxiv.org/abs/2210.03629) [[project]](https://react-lm.github.io/)
  - **Connection to your work:** You use DSPy's ReAct module - explain how it builds on this foundation

### 3.1.2 Fine-Tuning Methods

**Full Fine-Tuning**
- Traditional supervised fine-tuning on task-specific datasets
- High performance but computationally expensive
- OpenAI now supports GPT-4o fine-tuning ($25/1M tokens training)
- **Cite:** OpenAI Fine-Tuning Documentation [[docs]](https://platform.openai.com/docs/guides/fine-tuning/)

**Parameter-Efficient Fine-Tuning (PEFT): LoRA**
- Freezes base model, adds trainable low-rank matrices
- 10,000× fewer trainable parameters than full fine-tuning
- 3× reduction in GPU memory with no inference latency penalty
- **Cite:** Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models" [[arxiv]](https://arxiv.org/abs/2106.09685)
  - **Why this matters:** If fine-tuning smaller models (GPT-3.5), LoRA enables efficient training

**Direct Preference Optimization (DPO)**
- Optimizes models using pairwise comparisons
- Simpler than RLHF, no separate reward model needed
- Now supported for GPT-4.1 models (2025)
- **Cite:** OpenAI DPO Guide [[cookbook]](https://cookbook.openai.com/examples/fine_tuning_direct_preference_optimization_guide)

**When to use each approach (Table):**
| Method | Best For | Cost | Your Use Case |
|--------|----------|------|---------------|
| Few-shot prompting | Quick prototyping | Low | Initial baseline |
| DSPy compilation | Systematic optimization | Medium | Main approach |
| LoRA fine-tuning | Domain adaptation | Medium-High | If compilation insufficient |
| Full fine-tuning | Maximum performance | Very High | Not necessary for capstone |

---

## 3.2 DSPy Framework (1.5-2 pages) ⭐ **CORE SECTION**

### 3.2.1 Motivation: Why DSPy Exists

**The Prompting Problem:**
- Manual prompt engineering is brittle and doesn't scale
- Each pipeline change requires re-tuning all prompts
- No systematic way to optimize across model swaps
- **Cite:** Khattab et al. (2023) "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines" [[arxiv]](https://arxiv.org/abs/2310.03714) [[github]](https://github.com/stanfordnlp/dspy)

**DSPy's Solution:**
- Treat LM pipelines as **parameterized programs**
- Separate **what** to do (Signatures) from **how** to do it (Modules)
- **Compile** programs to optimize prompts/examples automatically

### 3.2.2 Key DSPy Concepts

**Signatures**
- Type signatures for LM transformations
- Example: `TaskBreakdownSignature` in your code
- Declare inputs/outputs, DSPy handles prompting

**Modules**
- Composable LM operations (Predict, ChainOfThought, ReAct)
- **Your usage:** `dspy.ReAct(TaskBreakdownSignature, tools=..., max_iters=15)`
- Modules are optimized during compilation

**Tools**
- Functions the LM can call during reasoning
- **Your tools:** `create_task_node`, `edit_task_node`, `update_task_status`, `delete_task_node`, `finish`
- Tools ground reasoning in actual graph operations

**Optimizers/Compilers**
- `BootstrapFewShot`: Generate demonstrations from training data
- `MIPRO`: Multi-prompt instruction optimization
- Automatically find better prompts + few-shot examples
- **Key result from paper:** Can outperform expert-written prompts in minutes

### 3.2.3 DSPy vs Other Frameworks

**DSPy vs LangChain**
- LangChain: Composable chains with fixed prompts
- DSPy: Optimizable programs with compiled prompts
- **Cite for LangChain context:** LangChain 1.0 documentation [[blog]](https://blog.langchain.com/langchain-langgraph-1dot0/)
  - LangChain good for rapid prototyping, DSPy better for optimization

**Why DSPy for your project:**
- Need consistent tool calling across complex reasoning
- Want to optimize task breakdown quality systematically
- Plan to fine-tune → DSPy traces provide training data

---

## 3.3 Agent Architectures and Tool Use (1 page)

### 3.3.1 Tool-Using LLM Agents

**Core Concept**
- LLMs generate function calls, external systems execute
- Enables grounding, up-to-date info, actionable outputs
- **General background** - cite ReAct paper again for tool use examples

### 3.3.2 Streaming Architectures

**Why Streaming Matters**
- User experience: see reasoning as it happens
- Incremental updates for long-running agents
- Transparency into agent decision-making
- **Your contribution:** Streaming DSPy ReAct with real-time graph updates

**Technical Challenges**
- Synchronous DSPy execution + Asynchronous FastAPI streaming
- **Your solution:** Thread-based queue (`backend/llm_model/agent.py:260-301`)
- Graph state consistency during streaming

**Recent Industry Examples**
- "Deep Agents" (Claude Code, Deep Research, Manus)
- Planning tools + sub-agents + file systems
- **Cite:** LangChain Deep Agents Blog [[blog]](https://blog.langchain.com/deep-agents/)

---

## 3.4 Task Decomposition and Planning Systems (1 page)

### 3.4.1 LLM-Based Task Planning

**Task Planning Categories** (from survey)
1. Task decomposition
2. Multi-plan selection
3. External planner integration
4. Reflection and self-critique
5. Memory systems
- **Cite:** "A Survey of Task Planning with Large Language Models" [[paper]](https://spj.science.org/doi/10.34133/icomputing.0124)

### 3.4.2 Adaptive Decomposition

**ADaPT (As-Needed Decomposition and Planning)**
- Recursively decomposes only when LLM can't execute directly
- 28.3% higher success on ALFWorld, 27% on WebShop
- **Cite:** Radhakrishnan et al. (2024) "ADaPT: As-Needed Decomposition and Planning with Language Models" [[arxiv]](https://arxiv.org/abs/2311.05772)

**Challenges with Graph Structures**
- LLMs struggle with task graph interpretation
- Sparse attention over graph structures
- Lack of graph isomorphism invariance
- **Your innovation:** Explicit graph operations via tools, not raw graph reasoning

### 3.4.3 Gap in Existing Work → Your Contribution

**What's missing:**
- No systematic optimization of task decomposition prompts (DSPy solves)
- Limited real-time streaming in planning agents (your architecture)
- No path from DSPy to fine-tuning for task graphs (your approach)

**Your unique contribution:**
- DSPy ReAct + streaming + graph tools + optimization path to fine-tuning

---

## Key Takeaways to Emphasize

1. **Prompting is not enough** → Need systematic optimization (DSPy)
2. **DSPy enables compilation** → But fine-tuning needed for domain specificity
3. **ReAct combines reasoning + acting** → Your foundation
4. **Task decomposition is hard** → Existing work lacks optimization + streaming
5. **Your gap** → Optimizable, streaming, graph-aware task breakdown system

---

## Writing Tips for Section 3

1. **Start each subsection with "why it matters"** before diving into details
2. **Use tables/figures** to compare approaches (prompting vs DSPy vs fine-tuning)
3. **Connect back to your work** - don't just survey, show relevance
4. **Be critical** - point out limitations that motivate your approach
5. **End with a clear gap statement** leading into your methodology

---

## Complete Source List

### Core Citations

1. **Wei et al. (2022)** - Chain-of-Thought Prompting
   https://arxiv.org/abs/2201.11903

2. **Yao et al. (2023)** - ReAct: Reasoning and Acting
   https://arxiv.org/abs/2210.03629
   Project: https://react-lm.github.io/

3. **Khattab et al. (2023)** - DSPy Framework ⭐ MOST IMPORTANT
   https://arxiv.org/abs/2310.03714
   GitHub: https://github.com/stanfordnlp/dspy

4. **Hu et al. (2021)** - LoRA: Low-Rank Adaptation
   https://arxiv.org/abs/2106.09685

5. **Radhakrishnan et al. (2024)** - ADaPT Task Decomposition
   https://arxiv.org/abs/2311.05772

### Documentation & Technical Resources

6. **OpenAI Fine-Tuning Docs**
   https://platform.openai.com/docs/guides/fine-tuning/

7. **OpenAI DPO Cookbook**
   https://cookbook.openai.com/examples/fine_tuning_direct_preference_optimization_guide

8. **LangChain 1.0 Release**
   https://blog.langchain.com/langchain-langgraph-1dot0/

9. **LangChain Deep Agents**
   https://blog.langchain.com/deep-agents/

### Survey Papers (Optional but recommended)

10. **"A Survey of Task Planning with Large Language Models"**
    https://spj.science.org/doi/10.34133/icomputing.0124

---

## Suggested Length Distribution

- 3.1 Language Model Optimization: **1.25 pages**
- 3.2 DSPy Framework: **2 pages** ⭐
- 3.3 Agent Architectures: **1 page**
- 3.4 Task Decomposition Systems: **1 page**
- **Total: ~5-5.5 pages** (slightly more than original plan, adjust as needed)

---

## Next Steps After Writing Section 3

1. Create comparison tables/figures
2. Add your own TaskBreakdownSignature as concrete DSPy example
3. Include diagram showing your architecture in context of related work
4. Write smooth transitions between subsections
5. End with 1-2 paragraph "gap summary" leading to Section 4
