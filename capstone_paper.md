# Optimizing Task Graph Generation through DSPy and Fine-Tuning: A Case Study in Intelligent Agent Systems

**Author:** [Your Name]
**Institution:** [Your Institution]
**Date:** December 2025

---

## Abstract

Task decomposition and management are fundamental challenges in project planning and execution, requiring intelligent systems that can reason about dependencies, hierarchies, and work breakdown structures. While large language models (LLMs) have demonstrated impressive capabilities in natural language understanding, their application to structured task graph generation remains challenging due to the need for consistent formatting, logical dependencies, and iterative refinement. This paper presents a novel approach to task graph generation using DSPy, a framework for programming—rather than prompting—language models, combined with strategic fine-tuning to optimize performance.

We develop an intelligent agent system that employs DSPy's ReAct (Reasoning and Acting) pattern to decompose complex user requests into hierarchical task graphs with proper parent-child relationships. Our system features real-time streaming updates, allowing users to observe the agent's reasoning process as task nodes are created, edited, and organized. We explore the transition from traditional prompt engineering to DSPy's programmatic optimization, demonstrating how systematic compilation and fine-tuning can improve task decomposition quality while reducing latency and cost.

Our experimental results show that DSPy optimization using BootstrapFewShot improves task breakdown accuracy by [X]% over baseline prompting, while fine-tuning GPT-4o-mini on domain-specific task graphs achieves [Y]% improvement with [Z]% cost reduction. We contribute: (1) a streaming architecture for real-time agent interaction, (2) a systematic evaluation framework for task decomposition quality, and (3) practical insights on when to employ DSPy compilation versus fine-tuning. This work has implications for production AI systems requiring structured outputs, multi-step reasoning, and optimization beyond manual prompt engineering.

**Keywords:** DSPy, Fine-tuning, ReAct Agents, Task Decomposition, Language Models, Streaming LLMs

---

## 1. Introduction

The rise of large language models (LLMs) has transformed how we approach complex reasoning tasks, from question answering to code generation. However, translating natural language requests into structured, actionable task graphs remains a significant challenge. Project management, software development workflows, and personal productivity systems all require intelligent decomposition of high-level goals into concrete, dependency-aware subtasks. While modern LLMs possess the reasoning capabilities to perform such decomposition, effectively harnessing these capabilities in production systems demands more than clever prompting.

Traditional approaches to LLM-powered task management rely on carefully crafted prompts, few-shot examples, and post-processing validation. These methods are brittle—small changes in phrasing can dramatically alter output quality, and maintaining consistency across tool calls, graph updates, and user interactions requires extensive manual engineering. Moreover, as task domains become more specialized (software engineering, research planning, event organization), generic LLM knowledge proves insufficient, necessitating domain adaptation through fine-tuning.

This paper addresses these challenges by leveraging **DSPy** (Declarative Self-improving Python), a framework that treats LM pipelines as parameterized programs rather than fixed prompts. Unlike traditional prompting frameworks such as LangChain, DSPy enables automatic optimization of prompts, demonstrations, and reasoning traces through compilation. We apply DSPy's ReAct module to task graph generation, creating an agent that systematically breaks down user requests while maintaining graph consistency through structured tool use. Our system streams updates in real-time, providing transparency into the agent's reasoning process—critical for user trust and iterative refinement.

Beyond DSPy optimization, we investigate when and how fine-tuning enhances performance for domain-specific task graphs. While DSPy's compilation can improve few-shot prompting, certain task patterns benefit from fine-tuning smaller models (GPT-4o-mini, GPT-3.5-turbo) on curated datasets of high-quality task decompositions. We develop evaluation metrics that capture task quality dimensions—completeness, logical dependencies, specificity, and graph validity—enabling systematic comparison across optimization strategies.

**Our contributions are threefold:**

1. **Streaming ReAct Agent Architecture**: We present a production-ready system that combines DSPy's ReAct pattern with asynchronous streaming, enabling real-time task graph updates as the agent reasons and acts. Our architecture (Section 4) demonstrates how to integrate DSPy with FastAPI for responsive user experiences.

2. **DSPy-to-Fine-tuning Pipeline**: We systematically explore the spectrum from prompt engineering to DSPy compilation to fine-tuning, providing practical guidance on when each approach is appropriate. Our experimental framework (Section 5-6) includes metrics for task quality, cost efficiency, and latency.

3. **Evaluation Framework for Task Decomposition**: We develop domain-specific metrics for assessing task graph quality beyond generic LLM benchmarks, including graph validity checks, dependency analysis, and user satisfaction measures.

The remainder of this paper is organized as follows: Section 2 reviews related work in LLM optimization and agent architectures. Section 3 provides background on DSPy and the ReAct pattern. Section 4 details our system design and implementation. Section 5 presents our optimization methodology from DSPy compilation to fine-tuning. Section 6 reports experimental results comparing approaches. Section 7 discusses findings and limitations. Section 8 concludes with implications for production AI systems.

---
