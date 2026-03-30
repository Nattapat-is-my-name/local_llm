# 🧠 Local LLM Course: Zero to Production

> Full Curriculum: ตั้งแต่พื้นฐานจนถึงการต่อ Backend จริง — เข้าใจทุก layer ของ LLM stack
>
> **Version 4.1 | March 2026 | Complete Edition**

---

## Table of Contents

- [Learning Path](#learning-path)
- [PART 0: Big Picture](#part-0-big-picture)
- [PART 1: What is Model](#part-1-what-is-model)
- [PART 2: Model Types](#part-2-model-types)
- [PART 3: xxB Model Size](#part-3-xxb-model-size)
- [PART 4: Model Format](#part-4-model-format)
- [PART 5: Quantization \& Precision](#part-5-quantization--precision)
- [PART 6: Runtime / Inference Engine](#part-6-runtime--inference-engine)
- [PART 7: GPU Framework](#part-7-gpu-framework)
- [PART 8: Hardware Selection](#part-8-hardware-selection)
- [PART 9: Run with LM Studio](#part-9-run-with-lm-studio)
- [PART 10: API Usage](#part-10-api-usage)
- [PART 11: Full Stack Integration](#part-11-full-stack-integration)
- [PART 12: Advanced Topics](#part-12-advanced-topics)
- [Bonus: DGX Spark \& UMA Deep Dive](#bonus-dgx-spark--uma-deep-dive)
- [Complete Formulas Reference](#complete-formulas-reference)

---

## Learning Path

```
0. Big Picture → ภาพรวมทั้งหมด
1. What is Model → LLM คืออะไร
2. Model Types → Base / Instruct / Embedding
3. xxB Model Size → 7B, 13B, 70B คืออะไร
4. Model Format → GGUF / Safetensors / ONNX
5. Quantization → FP16 → INT4
6. Runtime / Engine → llama.cpp / Ollama / vLLM
7. GPU Framework → CUDA / Metal / ROCm
8. Hardware Selection → เลือก hardware ให้เหมาะกับงาน
9. LM Studio → รันโมเดลแบบง่ายที่สุด
10. API Usage → เปิด API ใช้งานจริง
11. Full Stack → ต่อ Backend
12. Advanced → RAG, Scaling
```

---

# PART 0: Big Picture

## The LLM Stack Pipeline

ทุกครั้งที่ user พิมพ์คำถาม → ระบบต้องผ่านทุก layer ตามลำดับ

```
User → App/UI → Backend → Runtime → Model → Framework → Hardware
  1    2      3        4        5       6          7
```

| Layer | ตัวอย่าง | หน้าที่ |
|-------|----------|---------|
| **1. App** | Chat UI, Website, Slack Bot | รับ input จาก user แสดง output |
| **2. Backend** | Rails, Go, Node.js, Python | จัดการ logic, API, business rules |
| **3. Runtime** | llama.cpp, Ollama, vLLM | รันโมเดลจริง — inference engine |
| **4. Model** | GGUF, Safetensors, ONNX | ไฟล์โมเดลที่เก็บ weights |
| **5. Framework** | CUDA, Metal, ROCm, OpenCL | เชื่อม runtime → hardware |
| **6. Hardware** | NVIDIA GPU, Apple Silicon, CPU | คำนวณจริง |

### ภาพรวมการทำงาน

```
┌──────────────────────────────────────────────────────────────────┐
│                         USER INPUT                                │
│                     "What is AI?"                                 │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      1. APP / UI                                 │
│              Chat interface, CLI, API endpoint                     │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTPS / HTTP
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      2. BACKEND                                   │
│           Business logic, authentication, rate limiting            │
│                     Express/FastAPI/Rails                         │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTP to localhost
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      3. RUNTIME (llama.cpp)                      │
│        Inference engine — loads model, manages memory              │
│              Handles tokenization, generation                     │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      4. MODEL FILE (.gguf)                        │
│           Quantized weights, vocabulary, metadata                 │
│                  ~4-40 GB on disk                                │
└────────────────────────────┬─────────────────────────────────────┘
                             │ API calls
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      5. FRAMEWORK (CUDA/Metal)                   │
│         Low-level GPU APIs — memory management, kernels            │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      6. HARDWARE                                  │
│              NVIDIA GPU / Apple Silicon / CPU                      │
│                   Actual computation                             │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                       OUTPUT                                      │
│            "Artificial Intelligence (AI) is..."                    │
└──────────────────────────────────────────────────────────────────┘
```

> **จำไว้:** "User พิมพ์ → ไป Backend → ไป Runtime → ไป Model → ผ่าน Framework → บน Hardware → ได้ Output กลับมา"

---

# PART 1: What is Model

## LLM: Large Language Model

โมเดล AI ที่ถูก train ด้วยข้อมูล text จำนวนมหาศาล เพื่อทำ **Next Token Prediction**

### Core Concept — Neural Network คืออะไร?

ลองนึกภาพว่าสมองมนุษย์มี neuron หลายพันล้านตัว ที่เชื่อมต่อกัน แต่ละ connection มี "ความแข็งแกร่ง" (weight) ต่างกัน

AI model ก็เป็นตารางตัวเลข (weights) หลายพันล้านตัวที่บอกว่า neuron นี้ควร "ส่งสัญญาณ" ไปหา neuron ตัวไหนแค่ไหน

```
Input: "สวัสดี" 
  → Model weights (พันล้านตัวเลข)
  → Output: "สวัสดีครับ มีอะไรให้ช่วยไหม?"
```

### How LLM Works — Step by Step

```
Input: "The capital of France is"
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  1. TOKENIZATION                                           │
│     แปลง text → numbers (tokens)                           │
│     "The" → 464, "capital" → 1751, "of" → 3290            │
│     "France" → 319, "is" → 4886                            │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  2. EMBEDDING                                              │
│     แปลง token → vector (e.g., 4096 dimensions)             │
│     [464] → [0.234, -0.891, 0.442, ... 4096 dims]        │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  3. TRANSFORMER LAYERS                                      │
│     ประมวลผลผ่าน neural network หลาย layer                │
│                                                             │
│     ┌─────────────────────────────────────────────────┐     │
│     │  Layer N:                                        │     │
│     │  - Self-Attention: หาความสัมพันธ์ระหว่าง token │     │
│     │  - Feed-Forward: ประมวลผล patterns            │     │
│     │  - LayerNorm: normalize outputs                 │     │
│     └─────────────────────────────────────────────────┘     │
│                           │                                   │
│                           ▼                                   │
│     (ผ่าน 32-80+ layers ซ้ำๆ)                             │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  4. OUTPUT PROJECTION                                       │
│     แปลง hidden state → vocabulary logits                   │
│     [0.234, -0.891, ...] → [logits สำหรับทุก token ใน vocab]│
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  5. DECODING / SAMPLING                                     │
│     เลือก token ถัดไป (greedy, temperature, top-p)         │
│     "Paris" = token ที่มี probability สูงสุด                 │
└─────────────────────────────────────────────────────────────┘
```

### Two Phases of Inference

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM INFERENCE                             │
│                                                             │
│   ┌──────────────┐              ┌──────────────────────┐  │
│   │   PREFILL    │    ──────►    │       DECODE         │  │
│   │   PHASE      │              │       PHASE          │  │
│   └──────────────┘              └──────────────────────┘  │
│                                                             │
│   "What is AI?"        token-by-token generation            │
│   (one-shot)           until EOS / max tokens               │
│                                                             │
│   ⚡ FAST               🐢 SLOW (sequential)                │
│   parallelized          one token at a time                 │
└─────────────────────────────────────────────────────────────┘
```

#### **Prefill Phase** (Fast, Parallel)
- รับ prompt ทั้งหมด → ประมวลผลพร้อมกัน (parallel)
- สร้าง KV cache สำหรับทุก position ใน prompt
- ใช้เวลา: แปรผันตาม prompt length
- Output: "Time to First Token" (TTFT)

#### **Decode Phase** (Slow, Sequential)
- สร้างทีละ token (autoregressive)
- แต่ละ token ต้อง attention กับทุก token ก่อนหน้า
- ใช้ KV cache ที่สร้างไว้ใน prefill
- ใช้เวลา: แปรผันตาม total output length
- Output: "Time per Output Token" (TPOT)

## Vocabulary: ศัพท์ที่ต้องรู้

| คำศัพท์ | ความหมาย |
|---------|----------|
| **Token** | หน่วยเล็กที่สุดของ text (~0.75 คำ หรือ ~4 ตัวอักษร) |
| **Parameters** | ค่าน้ำหนักที่เรียนรู้ — 7B = 7 พันล้านค่า |
| **Context Window** | จำนวน token สูงสุดที่โมเดลรับได้ (เช่น 8K, 32K, 128K) |
| **Embedding** | แปลง token → vector ขนาดเช่น 4096 มิติ |
| **Vocabulary Size** | จำนวน token ที่โมเดลรู้จัก (เช่น 128,256 tokens) |
| **Layer** | ชั้นของ neural network — ยิ่งมาก = ยิ่ง deep |
| **Head** | Self-attention head — แต่ละ head ดูความสัมพันธ์ต่างกัน |
| **Logits** | คะแนนก่อน softmax — ยิ่งสูง = probability สูง |
| **Temperature** | ควบคุม randomness — สูง = 随机, ต่ำ = deterministic |
| **Top-p / Nucleus Sampling** | เลือกจาก smallest set ที่รวม probability = p |
| **EOS Token** | End of sequence — โมเดลบอกว่าจบแล้ว |

### Tokenization Examples

```
Text: "Hello, world!"

Tokenization (GPT-style):
  "Hello"    → 19973
  ","        → 11
  " world"   → 1917
  "!"        → 0 (usually)

Text: "ฉันรักAI" (Thai + English)

Tokenization:
  "ฉัน"     → 320
  "รัก"     → 1047
  "A"       → 32
  "I"       → 33

Note: English chars ~1 token per 4 chars
      Thai chars ~1-2 tokens per character
```

---

# PART 2: Model Types

แบ่งตามวัตถุประสงค์การใช้งาน

## Base Model
- **คือ:** โมเดลที่ train ด้วย text ทั่วไป (next token prediction)
- **ใช้งาน:** ต่อ sentence ที่เหมาะสม, text completion
- **ตัวอย่าง:** Llama-3.1-8B (base), Qwen-2.5-7B
- **ไม่เหมาะกับ:** Chat, Q&A โดยตรง

```
Input:  "The capital of France is"
Base:   "The capital of France is a beautiful city known for..."
```

## Instruct / Chat Model
- **คือ:** Base + fine-tune ให้ตอบคำถามได้ (Instruction Tuning)
- **ใช้งาน:** Chat, Q&A, งานทั่วไป
- **ตัวอย่าง:** Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3
- **การ train:** RLHF, DPO, SFT

```
Input:  "What is the capital of France?"
Instruct: "The capital of France is Paris."
```

## System / Completion Model
- **คือ:** Base + special tokens ให้รับ instruction ใน prompt
- **ตัวอย่าง:** CodeLlama, Qwen-Coder
- **ใช้งาน:** Code generation, fill-in-middle

## Embedding Model
- **คือ:** แปลง text → vector สำหรับค้นหา
- **ใช้งาน:** RAG, Semantic Search, Similarity
- **ตัวอย่าง:** nomic-embed-text, bge-large-en-v1.5
- **output:** Vector (e.g., 768, 1024, 1536 dimensions)

```
Input:  "What is AI?"
Embedding Model: → [0.234, -0.891, 0.442, ... 768 dims]

Similarity Search:
  "What is AI?"        → [0.234, -0.891, ...]  → Distance: 0.0
  "Tell me about ML"   → [0.231, -0.889, ...]  → Distance: 0.02
  "Weather today"      → [0.891, 0.442, ...]   → Distance: 0.85
```

## Multimodal Model
- **คือ:** รับได้หลาย modality — text + image + audio
- **ตัวอย่าง:** Llama-3.2-Vision, GPT-4V, Gemma-3-27B
- **ใช้งาน:** Image captioning, VQA, document understanding

> **สรุป:** ถ้าต้องการแชท → ใช้ Instruct model, ถ้าต้องทำ RAG → ใช้ Embedding model

---

# PART 3: xxB Model Size

## Parameters คืออะไร

Parameters คือ **weights (ค่าน้ำหนัก)** ที่เรียนรู้ระหว่าง training — ยิ่งเยอะ = ยิ่งเก่ง

```
7B  = 7,000,000,000 parameters
13B = 13,000,000,000 parameters
33B = 33,000,000,000 parameters
70B = 70,000,000,000 parameters
405B = 405,000,000,000 parameters
```

### Parameters ในโมเดลคืออะไรบ้าง?

```
Transformer Model Parameters Breakdown:

1. Embedding Layer
   - Token embeddings: vocab_size × hidden_dim
   - Example: 128,256 × 4096 = 525M parameters

2. Self-Attention (per layer)
   - Query projection:    hidden_dim × hidden_dim
   - Key projection:      hidden_dim × hidden_dim
   - Value projection:    hidden_dim × hidden_dim
   - Output projection:   hidden_dim × hidden_dim
   Total per layer: 4 × (d_model²)

3. Feed-Forward Network (per layer)
   - Up projection:       hidden_dim × ff_dim (usually 4× hidden_dim)
   - Down projection:     ff_dim × hidden_dim
   Total per layer: 2 × hidden_dim × ff_dim

4. LayerNorm (per layer)
   - 2 × hidden_dim (gamma + beta)

5. Output Layer
   - hidden_dim × vocab_size

Total: ~70% FFN + ~20% Attention + ~10% Embeddings
```

## Memory Requirements (FP16)

| Model Size | Parameters | FP16 RAM | Recommended Hardware |
|------------|------------|----------|---------------------|
| 1B-3B | 1-3 Billion | 2-6 GB | Laptop, Mac Mini |
| **7B** | 7 Billion | ~14 GB | RTX 3060+, M1 Pro+ |
| **13B** | 13 Billion | ~26 GB | RTX 4090, A100 |
| 30B | 30 Billion | ~60 GB | A100 40GB+ |
| 70B | 70 Billion | ~140 GB | Multi-GPU / Enterprise |
| 405B | 405 Billion | ~810 GB | Data center clusters |

## Real File Sizes (Quantized)

| Model Size | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q4_0 | Q3_K_M |
|-----------|------|------|------|--------|--------|-------|--------|
| 7B | 14 GB | 9.5 GB | 7.2 GB | 5.9 GB | 4.9 GB | 4.4 GB | 3.5 GB |
| 13B | 26 GB | 18 GB | 13 GB | 11 GB | 9 GB | 8.1 GB | 6.6 GB |
| 33B | 66 GB | 44 GB | 34 GB | 28 GB | 23 GB | 21 GB | 17 GB |
| 70B | 140 GB | 93 GB | 72 GB | 59 GB | 49 GB | 44 GB | 36 GB |
| 405B | 810 GB | 540 GB | 420 GB | 350 GB | 280 GB | 250 GB | 200 GB |

> **Trade-off:** โมเดลใหญ่กว่า = ดีกว่า แต่กิน RAM/VRAM มากกว่า
>
> **แนะนำ:** เริ่มจาก 7B ก่อน — เพียงพอสำหรับงานส่วนใหญ่

---

# PART 4: Model Format

"รูปแบบไฟล์ที่เก็บโมเดล" — แต่ละ format เหมาะกับ hardware ต่างกัน

## GGUF (GPT Generated Unified Format)
**Creator:** Georgi Gerganov (llama.cpp)

- **ดีที่สุดสำหรับ:** CPU inference, Apple Silicon (Metal), รัน local แบบง่าย
- **Extension:** .gguf
- **Quantization:** Q2-K to Q8_0 (built-in)
- **Features:**
  - Self-contained (model + tokenizer + metadata)
  - Easy to share and use
  - Fast quantization tools (llama-quantize)

### GGUF File Structure

```
┌─────────────────────────────────────────────┐
│  GGUF File                                  │
│  ┌─────────────────────────────────────────┐ │
│  │ Metadata (magic, version, architecture)  │ │
│  ├─────────────────────────────────────────┤ │
│  │ Hyperparameters (n_layers, n_heads...)  │ │
│  ├─────────────────────────────────────────┤ │
│  │ Vocabulary (tokenizer)                  │ │
│  ├─────────────────────────────────────────┤ │
│  │ Model Data (quantized weights)          │ │
│  │  - Tensor info (name, shape, offset)   │ │
│  │  - Binary weight data (Q4_K_M etc.)    │ │
│  └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

## Safetensors
**Creator:** HuggingFace

- **ดีที่สุดสำหรับ:** GPU inference (PyTorch), Training / Fine-tuning, Production deployment
- **Extension:** .safetensors
- **Quantization:** GPTQ, AWQ, ExL2, bitsandbytes
- **Features:**
  - Safe (no arbitrary code execution)
  - Fast loading (memory-mapped)
  - Lazy loading supported

## ONNX (Open Neural Network Exchange)
**Creator:** Microsoft

- **ดีที่สุดสำหรับ:** Cross-platform, Mobile / Edge, Web (WASM)
- **Extension:** .onnx
- **Tools:** ONNX Runtime, WinML, TensorRT
- **Features:**
  - Cross-framework compatibility
  - Hardware acceleration on edge devices
  - Web deployment via WebAssembly

## MLX
**Creator:** Apple

- **ดีที่สุดสำหรับ:** Apple Silicon (M1/M2/M3/M4)
- **Extension:** .mlx
- **Features:**
  - Unified memory architecture
  - Efficient quantization (MLX quantized)
  - Fast on Apple Silicon

## AWQ (Activation-aware Weight Quantization)
- **ดีที่สุดสำหรับ:** High quality 4-bit quantization
- **Extension:** .safetensors with AWQ
- **Features:**
  - Better than naive INT4
  - Requires less compute for same quality
  - Newer format, growing support

## Hardware Support Matrix

| Format | CPU | NVIDIA GPU | Apple Silicon | Mobile/Edge |
|--------|-----|------------|---------------|-------------|
| **GGUF** | ✅ ดีสุด | ✅ | ✅ Metal | 🟡 |
| **Safetensors** | 🟡 | ✅ ดีสุด | 🟡 | 🟡 |
| **ONNX** | ✅ | ✅ | 🟡 | ✅ ดีสุด |
| **MLX** | ❌ | ❌ | ✅ ดีสุด | ❌ |
| **AWQ** | 🟡 | ✅ | 🟡 | ❌ |

> **แนะนำสำหรับ Local LLM:**
> - NVIDIA GPU → GGUF หรือ Safetensors + GPTQ/AWQ
> - Apple Silicon → GGUF (Metal acceleration)
> - CPU Only → GGUF

---

# PART 5: Quantization & Precision

ลด precision เพื่อให้โมเดล "เบา" ลง — ลด RAM แต่กระทบความแม่นยำเล็กน้อย

## Precision คืออะไร?

Precision = จำนวน bits ที่ใช้เก็บแต่ละ parameter

```
FP32:  0.0034928174  (32 bits = 4 bytes) — Full precision
FP16:  0.00349       (16 bits = 2 bytes) — Half precision  
INT8:  42            (8 bits = 1 byte)  — 8-bit integer
INT4:  7             (4 bits = 0.5 bytes) — 4-bit integer
```

## Precision Types Deep Dive

### FP32 (Full Precision)
```
0.0034928174  (32 bits = 4 bytes)
```
- Precision เต็มที่สุด
- ใช้ VRAM มากที่สุด
- แทบไม่มีเหตุผลจะใช้สำหรับ inference

### FP16 (Half Precision)
```
0.00349  (16 bits = 2 bytes)
```
- คุณภาพใกล้เคียง FP32 มาก (~99.9%)
- มาตรฐานสำหรับ modern inference
- ทุก GPU รองรับ (Nvidia, AMD, Intel)

### BF16 (Brain Float 16)
```
0.00349  (16 bits = 2 bytes)
```
- สร้างมาเพื่อ AI โดยเฉพาะ (Google คิดค้น)
- Range กว้างกว่า FP16 → เหมาะกับ training
- คุณภาพเทียบเท่า FP16 แต่โมเดลส่วนใหญ่ยัง train ด้วย BF16

### INT8 (8-bit Integer)
```
42  (8 bits = 1 byte)
```
- ประหยัด 50% จาก FP16
- อาจมี quality loss เล็กน้อย (~1-3%)
- ใช้ quantization techniques พิเศษ

### INT4 (4-bit Integer)
```
7  (4 bits = 0.5 bytes)
```
- ประหยัด 75% จาก FP16
- Quality loss ชัดเจนขึ้น แต่ยังใช้ได้ (~5-15%)
- เหมาะกับ local LLM บนเครื่องที่มี RAM/VRAM จำกัด

## Precision Comparison Table

| Precision | ขนาด/param | คุณภาพ | VRAM/RAM (7B) | ใช้เมื่อ |
|-----------|-----------|-------|---------------|---------|
| **FP32** | 4 bytes | สูงสุด | ~28 GB | Production ที่ต้องการความแม่นยำเต็มที่ |
| **FP16** | 2 bytes | ดีมาก | ~14 GB | GPU ทั่วไป, เริ่มต้นที่ดี |
| **BF16** | 2 bytes | ดีมาก | ~14 GB | สร้างด้วย modern training |
| **INT8** (Q8_0) | 1 byte | ดี | ~7 GB | ต้องการประหยัดโดยไม่สูญเสียมาก |
| **INT6** (Q6_K) | 0.75 bytes | ดีมาก | ~5.2 GB | สมดุลดี |
| **INT5** (Q5_K_M) | 0.63 bytes | ดี | ~4.4 GB | แนะนำสำหรับ 8-12GB VRAM |
| **INT4** (Q4_K_M) | 0.56 bytes | พอใช้ | ~3.9 GB | แนะนำสำหรับ <8GB VRAM |
| **INT4** (Q4_0) | 0.47 bytes | ต่ำลง | ~3.3 GB | ระดับล่าง |
| **INT2** (Q2_K) | 0.28 bytes | ต่ำ | ~2 GB | ทดลอง, ไม่แนะนำใช้จริง |

## Quantization ทำงานยังไง

### Step-by-Step Process

```
Original FP16 weights: [0.7852, -0.1234, 0.5671, -0.8901, ...]
  │
  │ 1. Find scale and zero point
  │    scale = (max - min) / 127
  │    zero_point = round(-min / scale)
  │
  ▼
Quantized INT8: [100, -15, 72, -113, ...] (8-bit integers)
  │
  │ 2. Store: 1 byte per value (vs 2 bytes ก่อน)
  │
  ▼
Result: ลดขนาด 50% แต่ความแม่นยำลด ~1-3%
```

### Dequantization (during inference)

```
Quantized: 100
Scale: 0.00618
Zero point: 0

Dequantized: 100 × 0.00618 = 0.618 (close to original 0.7852)
```

### GGUF Quantization Types (Detailed)

| Type | Bits | 7B Size | Quality | แนะนำสำหรับ |
|------|------|---------|---------|--------------|
| **Q8_0** | 8 | ~7 GB | ~97% | RAM เยอะ ต้องการคุณภาพสูง |
| **Q6_K** | ~6 | ~5.5 GB | ~94% | Balance ระหว่างคุณภาพ+ขนาด |
| **Q5_K_M** | ~5 | ~4.9 GB | ~92% | แนะนำสำหรับ 8-12GB VRAM |
| **Q4_K_M ⭐** | ~4 | ~4.2 GB | ~89% | **Sweet spot — ส่วนใหญ่ใช้** |
| **Q4_0** | ~4 | ~3.8 GB | ~86% | VRAM จำกัดมาก |
| **Q3_K_M** | ~3 | ~3.5 GB | ~85% | VRAM จำกัดสุดๆ |
| **Q2_K** | ~2 | ~2.9 GB | ~80% | ทดลองเท่านั้น |

> **คำแนะนำ:** Q4_K_M คือ sweet spot สำหรับคนส่วนใหญ่ — ลดขนาด 70% แต่สูญเสียคุณภาพแค่ ~10%

### K_M vs K_S — ต่างกันยังไง?

- **Q4_K_M** = "Medium" — ใช้ extra tensor เก็บค่าที่สำคัญมากให้ precision สูงขึ้น
- **Q4_K_S** = "Small" — ประหยัดกว่าเล็กน้อย แต่คุณภาพน้อยกว่า
- **IQ4_XS** = "Improved Q4" — ดีกว่า Q4_K_M เล็กน้อยด้วย memory footprint ที่ใกล้เคียงกัน

**Q4_K_M แทบจะเป็น standard** ที่ใช้กันทั่วโลก

## Weight Inference Quantization — Deep Dive

### Core Concept

ระหว่าง **inference** โมเดลยังคงทำงานด้วย weights ที่ถูก quantized ไว้

```
Traditional:   FP16 weights (2 bytes each)
Quantized:      INT4 weights (0.5 bytes each) ← 4x smaller!
```

### Exact Process

```
1. Model trained in FP16/BF16 (full precision)
         ↓
2. Quantize weights → store as INT4/INT8
         ↓
3. During inference:
   - Load quantized weights (INT4)
   - Dequantize to FP16 on-the-fly for computation
   - Compute in FP16
   - Result: same output quality as FP16 inference
```

### Compute vs Storage Precision

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   STORAGE          →  QUANTIZED (INT4/INT8)   เพื่อประหยัด RAM│
│                                                             │
│   COMPUTE          →  ACTUAL calculation ใช้ FP16/BF16      │
│                                                             │
│   ═══════════════════════════════════════════════════════   │
│                                                             │
│   Weight file:  Q4_K_M (4-bit)    ← เก็บแบบ 4-bit          │
│   But when GPU calculates:  dequantize → FP16 → compute     │
│                                                             │
│   ∴ Quality ≈ FP16 (because compute is still high-prec)    │
│   But VRAM/RAM usage ≈ 4-bit file size                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## KV Cache Quantization — Deep Dive

### ทำไม KV Cache ถึงสำคัญมาก

KV Cache = **Memory ที่เก็บ Key และ Value tensors ของทุก token ที่ generate ไปแล้ว**

```
Prompt: "The capital of France is"
Tokens: [The] [capital] [of] [France] [is]
                 ↓
         KV Cache for each position:
         ─────────────────────────────────
         Position 1 (The):    K₁, V₁
         Position 2 (capital): K₂, V₂
         Position 3 (of):      K₃, V₃
         Position 4 (France):  K₄, V₄
         Position 5 (is):      K₅, V₅
         
         Next token: "Paris"
         → attention ต้อง เข้าถึง K₁-V₅ ทั้งหมด!
```

### KV Cache Memory Breakdown

**สำหรับโมเดล 7B, 4096 context length, FP16:**

```
Formula:
  KV_cache_size = 2 × n_layers × n_kv_heads × seq_len × head_dim × bytes_per_param

Breakdown:
  - n_layers = 32 (Llama 3.1 8B has 32 layers)
  - n_kv_heads = 8 ( Llama uses GQA, 8 KV heads vs 32 Q heads)
  - seq_len = 4096 (context length)
  - head_dim = 128 (for 4096 total dim with 32 heads)
  - bytes = 2 (FP16)

Calculation:
  = 2 × 32 × 8 × 4096 × 128 × 2 bytes
  = 536,870,912 bytes
  ≈ 512 MB per 4096 tokens!
```

### KV Cache Memory เทียบกับ Context Length

| Context Length | FP16 KV/1K tokens | INT8 KV/1K tokens | INT4 KV/1K tokens |
|----------------|-------------------|-------------------|-------------------|
| 4,096 | ~128 MB | ~64 MB | ~32 MB |
| 32,768 | ~1 GB | ~512 MB | ~256 MB |
| 131,072 | ~4 GB | ~2 GB | ~1 GB |

> **Warning:** KV cache ของ 128K tokens = 4 GB per sequence! ถ้ามี 3 concurrent users = 12 GB เฉพาะ KV cache

### KV Cache Quantization Types

#### **INT8 KV Cache**
```
Original KV:    FP16 = 2 bytes per value
Quantized KV:   INT8 = 1 byte per value

Memory savings: 50% reduction
Quality loss:   ~1-3% (usually acceptable)
Speed impact:   Minimal (dequantize on-the-fly for attention)
```

#### **FP8 KV Cache (E5M2 / E4M3)**
```
New format ที่ Nvidia H100 + Ada Lovelace รองรับ hardware
5-bit or 6-bit floating point
Better than INT8 for certain ranges

Speed:  เร็วกว่า INT8 (hardware-accelerated on modern GPUs)
Memory: ใกล้เคียง INT8
Quality: ดีกว่า INT8
```

### GQA vs MHA vs MQA — Attention Architecture

```
MHA (Multi-Head Attention) — Original
  - 32 Q heads, 32 K heads, 32 V heads
  - Heavy on KV cache
  - KV cache: 2 × 32 × 32 × seq_len × head_dim × bytes
  
MQA (Multi-Query Attention)
  - 32 Q heads, 1 K head, 1 V head (shared)
  - KV cache ลดลง 32x
  - Quality ลดลงบ้าง (เพราะ K,V ไม่แยกต่างหาก)
  
GQA (Grouped-Query Attention) — Standard now
  - 32 Q heads, 8 K heads, 8 V heads
  - สมดุลระหว่าง MHA และ MQA
  - KV cache ลดลง 4x จาก MHA
  - ใช้ใน Llama 3, Mistral, Qwen 2
```

### KV Cache Size by Attention Type

```
For Llama 3.1 8B:
  - 32 layers
  - 32 Q heads, 8 KV heads
  - head_dim = 128
  - seq_len = 4096
  - FP16

MHA KV cache = 2 × 32 × 32 × 4096 × 128 × 2 = 2 GB
GQA KV cache = 2 × 32 × 8  × 4096 × 128 × 2 = 512 MB
MQA KV cache = 2 × 32 × 1  × 4096 × 128 × 2 = 64 MB

GQA saves 4x KV cache vs MHA!
```

---

# PART 6: Runtime / Inference Engine

ตัวที่ "รันโมเดลจริง" — เลือก engine ให้เหมาะกับ use case

## llama.cpp
**Lightweight C/C++ Engine by Georgi Gerganov**

- **จุดเด่น:**
  - เร็วมากบน CPU (AVX2, AVX-512, ARM NEON)
  - รองรับ GGUF ทุก quantization
  - Apple Metal acceleration
  - ใช้ RAM หรือ VRAM ก็ได้
  - GPU offloading (layer-based)
  - Cross-platform (Linux, macOS, Windows)
- **ใช้เมื่อ:** Local inference, CPU/GPU mixed, embedded systems
- **Used by:** LM Studio, Ollama (under the hood), text-generation-webui

### llama.cpp Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    llama.cpp                                 │
│                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │  gguf.c     │   │  llama.cpp  │   │  ɗ-common   │       │
│  │  (format)   │ → │  (core)     │ → │  (util)     │       │
│  └─────────────┘   └──────┬──────┘   └─────────────┘       │
│                           │                                  │
│                           ▼                                  │
│                    ┌─────────────┐                          │
│                    │  Backend    │                          │
│                    │  CUDA / Metal│                          │
│                    │  / CPU /    │                          │
│                    │  OpenCL     │                          │
│                    └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

## Ollama
**User-Friendly Wrapper around llama.cpp**

- **จุดเด่น:**
  - ใช้ง่ายมาก: `ollama run llama3`
  - มี CLI สวยงาม
  - Built-in API server
  - Model library พร้อมใช้ (ollama.com/library)
  - Supports Modelfile (customization)
  - Cross-platform
- **ใช้เมื่อ:** Dev/Testing อย่างรวดเร็ว, quick prototyping
- **จุดด้อย:** Less customizable than raw llama.cpp

### Ollama Commands

```bash
# Run a model
ollama run llama3.1:8b-instruct-q4_0

# List installed models
ollama list

# Create custom model (Modelfile)
ollama create mymodel -f Modelfile

# API server
ollama serve
```

## vLLM
**Production-Grade Engine by UC Berkeley**

- **จุดเด่น:**
  - **PagedAttention** — KV cache management แบบ virtual memory
  - Throughput สูงมาก (2-5x vs naive implementation)
  - **Continuous batching** — maximizes GPU utilization
  - **Tensor parallelism** — scale across multiple GPUs
  - OpenAI-compatible API
- **ใช้เมื่อ:** Production, multi-user, high-throughput scenarios
- **จุดด้อย:** GPU-heavy (no CPU inference), more complex setup

### vLLM vs llama.cpp vs Ollama

| Criteria | llama.cpp | Ollama | vLLM |
|----------|-----------|--------|------|
| **Ease of Use** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Speed (Latency)** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Throughput** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **CPU Support** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
| **Multi-GPU** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Best For** | Local, Custom | Dev, Testing | Production |

> **ความเข้าใจสำคัญ:**
> - llama.cpp = **Engine** (ตัวประมวลผล)
> - Ollama = **Wrapper + UX** (ห่อ llama.cpp ให้ใช้ง่าย)
> - vLLM = **Server Engine** (สำหรับ production scale)

## Throughput vs Latency

| | Latency | Throughput |
|--|---------|------------|
| **คือ** | เวลาที่รอก่อนได้รับ response แรก | จำนวน request ที่ประมวลผลได้ต่อวินาที |
| **เหมาะกับ** | ผู้ใช้คนเดียว, Interactive use, Chatbot | ผู้ใช้หลายคนพร้อมกัน, Batch processing, API service |
| **แนะนำ** | llama.cpp | vLLM |

---

# PART 7: GPU Framework

ตัวเชื่อม Runtime → Hardware — เลือกไม่ได้เลือก hardware ไม่ได้

## CUDA
**NVIDIA Only**

- Ecosystem ใหญ่ที่สุด
- ทุก AI framework รองรับ (PyTorch, TensorFlow, JAX)
- Driver ดีที่สุด
- Performance สูงสุด
- ใช้กับ: NVIDIA GPU เท่านั้น (RTX, A100, H100, L40)

### CUDA Versions

| CUDA Version | Min Driver | GPU Support |
|-------------|-----------|-------------|
| CUDA 11.8 | 470.x | Ampere, Ada, Hopper |
| CUDA 12.x | 525.x | Ada, Hopper, Blackwell |

## Metal
**Apple Silicon**

- Apple M-Series สำหรับ Mac
- Unified Memory Architecture
- Power efficient
- llama.cpp (Metal acceleration), MLX acceleration
- ใช้กับ: Mac เท่านั้น

## ROCm
**AMD GPU (Linux)**

- Open-source สำหรับ AMD
- PyTorch, JAX support
- ยัง比不上 CUDA ecosystem
- ใช้กับ: AMD GPU (RX 7000 series, MI200, MI300)

## OpenCL
**Cross-Platform**

- Open standard
- Less optimized than CUDA/Metal
- Fallback option
- ใช้กับ: Older GPUs, cross-vendor

## Hardware → Framework Mapping

| Hardware | Framework | Engine ที่รองรับ |
|----------|-----------|------------------|
| NVIDIA GPU (RTX, A100, H100) | **CUDA** | llama.cpp, vLLM, text-generation-webui, Ollama |
| Apple Silicon (M1-M4) | **Metal** | llama.cpp (Metal), Ollama, MLX |
| AMD GPU | **ROCm** | llama.cpp (ROCm build), Ollama |
| CPU Only | **BLAS** (OpenBLAS/MKL) | llama.cpp (BLAS), Ollama |

> **สำคัญ:** Framework ต้องตรงกับ Hardware!
> - NVIDIA GPU → ต้องใช้ CUDA
> - Mac → ต้องใช้ Metal
> - ผิด framework = รันไม่ได้

---

# PART 8: Hardware Selection

Rule หลัก: RAM/VRAM ต้อง > ขนาดโมเดลที่quantize แล้ว

## Memory Formula

```
Memory = Model_Params × Bytes_per_Param + KV_Cache + Overhead

Where:
  - FP16: 2 bytes per param
  - Q8_0: 1 byte per param
  - Q6_K: 0.75 bytes per param
  - Q5_K_M: 0.63 bytes per param
  - Q4_K_M: 0.56 bytes per param
  
  KV_Cache = 2 × n_layers × n_kv_heads × seq_len × head_dim × bytes
```

## Quick Hardware Guide

| Hardware | Memory | รันได้ | Speed | ราคาโดยประมาณ |
|----------|--------|--------|-------|---------------|
| Mac Mini M4 | 16-32 GB | 3B-7B Q4 | Medium | ~$600-1200 |
| Mac Studio M3 Max | 64-128 GB | 13B-70B Q4 | Fast | ~$3000-8000 |
| RTX 3060 | 12 GB VRAM | 7B Q4 | Fast | ~$300 |
| RTX 4080 | 16 GB VRAM | 13B Q4 | Faster | ~$1200 |
| RTX 4090 | 24 GB VRAM | 30B Q4 | Fastest (Consumer) | ~$1800 |
| A100 40GB | 40 GB VRAM | 30B-70B Q4 | Very Fast | ~$10,000+ |
| A100 80GB | 80 GB VRAM | 70B FP16 | Extremely Fast | ~$15,000+ |
| DGX Spark | 128 GB RAM | 70B Q4 (NVFP4) | Very Fast (Pre-fill) | ~$3,000 |

## GPU Recommendations

### แบบ Professional / Serious Use

| GPU | VRAM | ราคาเป็น | เหมาะกับ |
|-----|------|---------|---------|
| **RTX 4090** | 24 GB | High | รัน 70B Q4, 33B FP16, ทำ everything |
| **A100 40GB** | 40 GB | Very High | Datacenter, รัน 70B FP16 ได้ |
| **A100 80GB** | 80 GB | Very High | Production, 70B FP16, fine-tuning |
| **A6000** | 48 GB | Very High | คล้าย A100 แต่ Ada generation |
| **RTX 3090** | 24 GB | Mid-High | รัน 33B Q4 สบาย, 70B Q4 ก็ได้ |
| **RTX 4080 Super** | 16 GB | High | รัน 13B FP16, 34B Q4 สบาย |

### แบบ Budget / ประหยัด

| GPU | VRAM | ราคา | เหมาะกับ |
|-----|------|------|---------|
| **RTX 4060 Ti 16GB** | 16 GB | Mid | รัน 13B FP16, 34B Q4, ราคาดี |
| **RTX 3060 12GB** | 12 GB | Low | รัน 13B Q6, 7B FP16 |
| **RTX 4060 8GB** | 8 GB | Low | รัน 7B Q6, 13B Q4 |
| **Arc B580 12GB** | 12 GB | Mid | ทางเลือก Intel, เร็ว, ราคาดี |

### Apple Silicon (Mac)

| Chip | Unified Memory | เหมาะกับ |
|------|---------------|---------|
| **M1 Max** | 64 GB | รัน 34B Q4 สบาย |
| **M2 Max** | 64 GB | M1 Max แต่เร็วกว่า 20-30% |
| **M3 Max** | 128 GB | รัน 70B Q4 ได้ |
| **M4 Max** | 128 GB | M3 Max แต่เร็วกว่า 15-20% |
| **M3 Pro 36GB** | 36 GB | รัน 13B Q6, 34B Q4 |

## Memory Calculator (7B Model)

| Quantization | Weights | +4K KV | +8K KV | +32K KV | +128K KV |
|-------------|---------|--------|--------|---------|----------|
| FP16 | 14 GB | 14.25 GB | 14.5 GB | 16 GB | 24 GB |
| Q8_0 | 7 GB | 7.25 GB | 7.5 GB | 9 GB | 17 GB |
| Q6_K | 5.5 GB | 5.75 GB | 6 GB | 7.5 GB | 15.5 GB |
| Q5_K_M | 4.9 GB | 5.15 GB | 5.4 GB | 6.9 GB | 14.9 GB |
| Q4_K_M | 4.2 GB | 4.45 GB | 4.7 GB | 6.2 GB | 14.2 GB |

## Use Case → Hardware แนะนำ

### บทเรียน / เริ่มต้น
- **Mac Mini M4 24GB:** $999 — เพียงพอสำหรับ 7B
- **RTX 3060 12GB:** $300 — ถูกที่สุดที่ใช้ได้

### Developer / Production
- **RTX 4090 24GB:** $1800 — best value performance
- **Mac Studio 128GB:** $8000 — unified memory เยอะ

### Team / Enterprise
- **DGX Spark × 2:** $6000 — 256GB, scalable
- **A100 80GB × 2:** $20k+ — production grade

### Fine-tuning / Research
- **DGX Spark:** Fine-tune ได้ในเครื่อง
- **A100/H100:** Training + Inference

## Layer Offloading — GPU + CPU ทำงานด้วยกัน

นี่คือเทคนิคสำคัญที่ทำให้ **VRAM ไม่พอ แต่อยากรันโมเดลใหญ่**

### Concept

โมเดล 7B มี 33 layers (transformer layers)

```
Layer 1 ──► Layer 2 ──► ... ──► Layer 33
   │          │                 │
 GPU:15      GPU:15          GPU:3 + CPU:30
 layers     layers           layers
```

- ปกติ: แต่ละ layer ต้อง load ใน VRAM
- Layer offloading: โหลดแค่บาง layer ไป GPU, ที่เหลืออยู่ CPU

### ตัวอย่าง

```
Model: Llama 3.1 8B Q4_K_M (~4.5 GB)
GPU: RTX 3060 12GB

แต่ถ้า model 70B Q4 (~40 GB):
- GPU VRAM: 12 GB → load ได้แค่ 12/40 = ~30% = 21 layers
- CPU: รัน 21 layers ที่เหลือ

→ ทำให้รัน 70B ได้แม้ VRAM แค่ 12GB แต่... ช้ามาก
```

---

# PART 9: Run with LM Studio

วิธีง่ายที่สุดในการรัน Local LLM — ไม่ต้องใช้ command line

## Installation

```bash
# macOS / Windows / Linux
1. ไปที่ https://lmstudio.ai
2. ดาวน์โหลด version สำหรับ OS ของคุณ
3. ติดตั้งและเปิดโปรแกรม
4. ดาวน์โหลดโมเดลที่ต้องการ
5. โหลดโมเดล → แชทได้เลย!
```

## Step by Step

### 1. ดาวน์โหลดโมเดล
```
1. Click 🔍 (Search models)
2. ค้นหา "Llama-3.1-8B-Instruct-GGUF"
3. เลือก Q4_K_M version
4. Click Download
```

### 2. โหลดโมเดล
```
1. ไปที่ Model tab (sidebar)
2. Click ที่โมเดลที่ดาวน์โหลดมา
3. รอจนโหลดเสร็จ (VRAM usage ขึ้น)
```

### 3. ปรับ GPU Offload
```
Slider: "GPU Layers" หรือ "Offload"
- เลื่อนไปทางขวา = ใช้ GPU มากขึ้น = เร็วกว่า
- เลื่อนไปทางซ้าย = ใช้ RAM มากขึ้น = VRAM ประหยัดกว่า

แนะนำ: เลื่อนไป max ก่อน ถ้า OOM แล้วค่อยลด
```

### 4. เริ่มแชท
```
1. ไปที่ Chat tab
2. พิมพ์คำถาม
3. Enter
```

## LM Studio Interface Overview

```
┌─────────────────────────────────────────────────────────────┐
│  LM Studio                                                 │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                      │
│  │  🔍     │ │   💬    │ │   ⚙️    │                      │
│  │ Search  │ │  Chat   │ │Settings │                      │
│  └─────────┘ └─────────┘ └─────────┘                      │
│                                                             │
│  [Model loaded: Llama-3.1-8B-Instruct-Q4_K_M]              │
│  GPU: ████████░░ 12GB/16GB                                 │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ User: What is machine learning?                     │   │
│  │                                                      │   │
│  │ Assistant: Machine learning is a subset of AI...     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Temperature: 0.7  │  Max Tokens: 2048  │  Context: 8K    │
└─────────────────────────────────────────────────────────────┘
```

## Advanced Settings

| Setting | Description | Recommended |
|---------|-------------|-------------|
| **Temperature** | Controls randomness (0 = deterministic, 1 = creative) | 0.7 for balanced |
| **Max Tokens** | Maximum output length | 2048-4096 |
| **Context Length** | Maximum input + output tokens | Model dependent |
| **GPU Layers** | How many layers to offload to GPU | Max possible |
| **Batch Size** | Tokens processed in parallel | 512 for speed |
| **Top-P** | Nucleus sampling threshold | 0.9-0.95 |
| **Repeat Penalty** | Penalize repeated tokens | 1.1-1.2 |

---

# PART 10: API Usage

LM Studio มี built-in OpenAI-compatible API — ใช้แทน OpenAI ได้เลย

## Start API Server

```
1. ไปที่ tab "API" ใน LM Studio
2. Click "Start Server"
3. Server รันที่ http://localhost:1234/v1/chat/completions
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (main endpoint) |
| `/v1/completions` | POST | Text completion |
| `/v1/models` | GET | List available models |
| `/v1/embeddings` | POST | Get embeddings |

## API Call Example

### Chat Completion

```bash
curl http://localhost:1234/v1/chat/completions \
 -H "Content-Type: application/json" \
 -H "Authorization: Bearer no-key" \
 -d '{
 "model": "local-model",
 "messages": [
   {"role": "system", "content": "You are a helpful assistant."},
   {"role": "user", "content": "Hello!"}
 ],
 "temperature": 0.7,
 "max_tokens": 512,
 "stream": false
 }'
```

### Response

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "local-model",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 12,
    "total_tokens": 27
  }
}
```

## Code Integration

### Python — OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="no-key",
    base_url="http://localhost:1234/v1"  # ต่อ local แทน OpenAI
)

response = client.chat.completions.create(
    model="local-model",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    temperature=0.7,
    max_tokens=1000
)

print(response.choices[0].message.content)
```

### Python — Requests Library

```python
import requests

response = requests.post(
    "http://localhost:1234/v1/chat/completions",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer no-key"
    },
    json={
        "model": "local-model",
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

### JavaScript — Node.js

```javascript
const response = await fetch('http://localhost:1234/v1/chat/completions', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer no-key'
    },
    body: JSON.stringify({
        model: 'local-model',
        messages: [
            { role: 'user', content: 'Hello!' }
        ],
        temperature: 0.7,
        max_tokens: 512
    })
});

const data = await response.json();
console.log(data.choices[0].message.content);
```

> **จุดเด่น:** เปลี่ยนแค่ base_url กับ api_key — ใช้ code เดิมที่เคยใช้กับ OpenAI ได้เลย!

---

# PART 11: Full Stack Integration

ต่อ Local LLM เข้ากับ Backend จริง — ทำให้เป็น API ของตัวเอง

## System Architecture

```
┌──────────┐
│ Frontend │ React / Vue / Mobile App / Discord Bot
└────┬─────┘
     │ HTTPS / WebSocket
     ▼
┌──────────┐         ┌──────────────────────────────┐
│ Backend  │─────────►│  LLM Server (LM Studio/Ollama) │
│ (Your)   │ HTTP    │  http://localhost:1234/v1     │
└────┬─────┘         └──────────────────────────────┘
     │
     ▼
┌──────────┐
│ Database │ PostgreSQL / MongoDB / SQLite
│ (Logs)   │
└──────────┘
```

## Backend Examples

### Node.js / Express

```javascript
const express = require('express');
const { OpenAI } = require('openai');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

const client = new OpenAI({
    apiKey: 'no-key',
    baseURL: 'http://localhost:1234/v1'
});

// Store conversation history
const conversations = new Map();

app.post('/chat', async (req, res) => {
    try {
        const { userId, message, systemPrompt } = req.body;
        
        // Get or create conversation history
        if (!conversations.has(userId)) {
            conversations.set(userId, []);
        }
        const history = conversations.get(userId);
        
        // Build messages
        const messages = [
            ...(systemPrompt ? [{ role: 'system', content: systemPrompt }] : []),
            ...history,
            { role: 'user', content: message }
        ];
        
        const response = await client.chat.completions.create({
            model: 'local-model',
            messages,
            temperature: 0.7,
            max_tokens: 1000
        });
        
        const reply = response.choices[0].message.content;
        
        // Update history
        history.push({ role: 'user', content: message });
        history.push({ role: 'assistant', content: reply });
        
        // Keep history manageable
        if (history.length > 20) {
            history.splice(0, 4); // Remove oldest 2 turns
        }
        
        res.json({
            reply,
            usage: response.usage,
            conversationId: userId
        });
    } catch (error) {
        console.error('Chat error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Reset conversation
app.post('/chat/:userId/reset', (req, res) => {
    const { userId } = req.params;
    conversations.delete(userId);
    res.json({ success: true });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`LLM Server running on port ${PORT}`));
```

### Python / FastAPI

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import openai
import uuid

app = FastAPI()

openai.api_key = "no-key"
openai.api_base = "http://localhost:1234/v1"

# Store conversations
conversations = {}

class ChatRequest(BaseModel):
    user_id: str
    message: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000

class ResetRequest(BaseModel):
    user_id: str

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        # Get or create conversation
        if req.user_id not in conversations:
            conversations[req.user_id] = []
        history = conversations[req.user_id]
        
        # Build messages
        messages = []
        if req.system_prompt:
            messages.append({"role": "system", "content": req.system_prompt})
        messages.extend(history)
        messages.append({"role": "user", "content": req.message})
        
        # Call LLM
        response = openai.ChatCompletion.create(
            model="local-model",
            messages=messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens
        )
        
        reply = response.choices[0].message.content
        
        # Update history
        history.append({"role": "user", "content": req.message})
        history.append({"role": "assistant", "content": reply})
        
        # Keep history manageable
        if len(history) > 20:
            history[:] = history[-16:]
        
        return {
            "reply": reply,
            "usage": response.usage,
            "conversation_id": req.user_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/reset")
async def reset_conversation(req: ResetRequest):
    if req.user_id in conversations:
        del conversations[req.user_id]
    return {"success": True, "message": "Conversation reset"}

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "local-model"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
```

### Adding Authentication

```python
from fastapi import FastAPI, HTTPException, Depends, Header
from typing import Optional

app = FastAPI()

# Simple API key validation
API_KEYS = {
    "user1_key": "user1",
    "user2_key": "user2"
}

def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return API_KEYS[x_api_key]

@app.post("/chat")
async def chat(req: ChatRequest, user_id: str = Depends(verify_api_key)):
    # Now user_id comes from the API key
    # ... rest of chat logic
```

### Adding Rate Limiting

```python
from fastapi import FastAPI, HTTPException
from ratelimit import rate_limit
from time import time

# Simple in-memory rate limiter
rate_limits = {}

def check_rate_limit(user_id: str, limit: int = 60, window: int = 60) -> bool:
    now = time()
    if user_id not in rate_limits:
        rate_limits[user_id] = []
    
    # Remove old requests
    rate_limits[user_id] = [
        t for t in rate_limits[user_id]
        if now - t < window
    ]
    
    if len(rate_limits[user_id]) >= limit:
        return False
    
    rate_limits[user_id].append(now)
    return True

@app.post("/chat")
async def chat(req: ChatRequest):
    if not check_rate_limit(req.user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    # ... rest of chat logic
```

## Production Considerations

| Aspects | สิ่งที่ต้องคิด |
|---------|---------------|
| **Authentication** | เพิ่ม API key หน้า backend ของตัวเอง |
| **Rate Limiting** | ป้องกัน user ส่ง request เยอะเกิน |
| **Caching** | Cache response ที่ถามซ้ำๆ (semantic cache) |
| **Concurrent Users** | vLLM รองรับได้ดีกว่า llama.cpp |
| **Monitoring** | Log token usage, latency, errors |
| **Scaling** | Horizontal scaling with load balancer |
| **Fallback** | Fallback to cloud API if local fails |

---

# PART 12: Advanced Topics

เนื้อหาขั้นสูงสำหรับ production และ scaling

## RAG: Retrieval Augmented Generation

RAG = ดึงข้อมูลจาก external knowledge base มาต่อด้วย context

```
User: "ผลงานล่าสุดของบริษัทคืออะไร?"
 │
 ▼
┌─────────────────────────────────────────────────────────────┐
│  1. EMBEDDING                                              │
│     แปลงคำถาม → vector                                     │
│     "ผลงานล่าสุดของบริษัทคืออะไร?"                         │
│      → [0.234, -0.891, 0.442, ... 1536 dims]              │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  2. RETRIEVAL                                              │
│     ค้นหา docs ที่เกี่ยวข้องจาก vector DB                  │
│                                                             │
│     ChromaDB / Pinecone / Weaviate / Milvus                 │
│      ↓                                                     │
│     "Q3 2026 product launch: AI Assistant Pro"             │
│     "New partnerships with TechCorp"                        │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  3. AUGMENT                                                │
│     ใส่ context เข้าไปใน prompt                            │
│                                                             │
│     System: "ตอบโดยอิงจาก context ที่ให้"                  │
│     Context: "[1] Q3 2026: เปิดตัว AI Assistant Pro..."    │
│     Question: "ผลงานล่าสุดของบริษัทคืออะไร?"              │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  4. GENERATE                                               │
│     ส่งไปให้ LLM ตอบ                                       │
│     → "ผลงานล่าสุดคือการเปิดตัว AI Assistant Pro ใน Q3..." │
└─────────────────────────────────────────────────────────────┘
```

### RAG Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG SYSTEM                               │
│                                                             │
│  ┌───────────────┐      ┌─────────────────┐                  │
│  │   Document    │      │   Embedding     │                  │
│  │   Loader     │ ───► │   Model         │                  │
│  │ (PDF, URL)   │      │ (nomic-embed)   │                  │
│  └───────────────┘      └────────┬────────┘                  │
│                                 │                           │
│                                 ▼                           │
│                        ┌─────────────────┐                  │
│                        │  Vector Store   │                  │
│                        │ (ChromaDB etc.) │                  │
│                        └────────┬────────┘                  │
│                                 │                           │
│                                 ▼                           │
│  ┌───────────────┐      ┌─────────────────┐                  │
│  │    User       │      │   Retrieval      │                  │
│  │    Query      │ ───► │   (top-k)       │                  │
│  └───────────────┘      └────────┬────────┘                  │
│                                 │                           │
│                                 ▼                           │
│                        ┌─────────────────┐                  │
│                        │     LLM         │                  │
│                        │  (Local Model)  │                  │
│                        └─────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

### RAG Tools

| Component | Tools |
|-----------|-------|
| **Embedding Model** | nomic-embed-text, bge-large-en-v1.5, text-embedding-ada-002 |
| **Vector Database** | Pinecone, Weaviate, ChromaDB, Milvus, Qdrant |
| **RAG Framework** | LangChain, LlamaIndex (formerly GPT-Index), Haystack |
| **Document Loader** | PyPDF, Unstructured, Playwright (web) |

## KV Cache Quantization (Advanced)

นอกจาก weights แล้ว KV cache ก็ quantization ได้

| KV Cache Precision | Memory Saved | Quality Impact | When to Use |
|-------------------|--------------|----------------|-------------|
| FP16 (none) | 0% | None | VRAM เยอะ |
| INT8 | 50% | Minimal (~1-3%) | Long context |
| INT4 | 75% | Small (~3-5%) | Very long context |
| FP8 (E5M2) | 50% | Very small | H100/Ada |

> **ความสำคัญ:** KV cache ของ 32K context = ~8x ของ 4K context!
> ถ้าต้องการ long context → ต้อง quantization KV cache ด้วย

## Scaling: Multi-GPU & Distributed

### Tensor Parallelism
- แบ่งโมเดลไปหลาย GPU (ภายในเครื่องเดียว)
- vLLM: `--tensor-parallel-size=2`
- สำหรับ 70B+ models

### Pipeline Parallelism
- แบ่ง layers ไปหลาย machine
- DeepSeek, LlamaStack
- ข้าม machine ได้

### Data Parallelism
- เหมาะกับ Training ไม่ใช่ Inference
- Multiple copies of model on different GPUs
- Process different batches in parallel

## Context Length & Optimization

| Context | Tokens | Use Case |
|---------|--------|----------|
| Short | 2K-4K | Chat ทั่วไป |
| Medium | 8K-16K | Code review, เอกสารยาว |
| Long | 32K-128K | หนังสือ, codebase ใหญ่ |
| Extended | 200K+ | Research, analysis |

### Position Encoding Extensions

```
Problem:  Model ถูก train ด้วย context เช่น 4096 tokens
          แต่อยากใช้ 128K tokens?

Solutions:

1. Positional Interpolation (PI)
   - Scale existing position indices
   - 4096 trained → now covers 128K
   - Works well for moderate extension (4x)
   
2. YaRN (Yet another RoPE extensioN)
   - More aggressive extension
   - Better quality at extreme lengths
   - Recommended for 100K+ contexts
   
3. NTK-aware Scaling
   - Dynamic changes to attention base
   - Minimal quality loss
```

> **Memory Warning:** KV cache ของ 32K = 8x ของ 4K (ทั้ง compute + storage)

## Flash Attention — เร็วขึ้น 2-4x

**Problem with naive attention:**

```
Standard attention:
  - O(n² × d) memory complexity
  - For n=4096, d=128:  ต้อง allocate huge intermediate matrix
  - ทำให้ VRAM หมดเร็ว
  
Standard:  S = Q × K^T  (n×n matrix = 16M elements for n=4096!)
          P = softmax(S)  (16M)
          O = P × V      (16M)
          
Total memory for n=4096:  ~256 MB just for attention scores
```

**Flash Attention Solution:**

```
Algorithm: tiles the attention computation into smaller blocks
         that fit in SRAM/L1 cache

1. ไม่ต้องสร้าง full n×n matrix
2. คำนวณทีละ block, accumulate results
3. Memory: O(n) แทน O(n²)

Speed:  2-4x faster
Memory: 10-20x less

Trade-off:  slight quality loss due to numerical precision in tiling
```

## Generation Speed — TTFT, TPOT, Throughput

### Key Metrics

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  TTFT  = Time To First Token                                │
│          เวลาที่รอ token แรกหลังกด enter                    │
│          วัด prefill speed                                  │
│                                                             │
│  TPOT  = Time Per Output Token                              │
│          เวลาต่อ token ที่ generate ออกมา                  │
│          วัด decode speed                                  │
│          ยิ่งต่ำ = ยิ่งเร็ว                                │
│                                                             │
│  E2EL  = End-to-End Latency                                 │
│          TTFT + (TPOT × num_output_tokens)                  │
│          เวลาทั้งหมดจนได้ response                         │
│                                                             │
│  Thrp  = Throughput (tokens/second)                         │
│          รวมทุก sequences                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### TTFT (Time To First Token) — Example

**ส่วนประกอบ:**

```
TTFT = Tokenization time
     + Model loading to GPU (ถ้ายังไม่ loaded)
     + Prefill time (prompt processing)
     + Queue wait (ถ้ามี concurrent requests)
```

**TTFT ตัวอย่าง (RTX 4090, Llama 3.2 8B Q4_K_M):**

| Prompt Length | TTFT |
|---------------|------|
| 100 tokens | ~50-100 ms |
| 500 tokens | ~200-400 ms |
| 2,000 tokens | ~800-1500 ms |
| 8,000 tokens | ~3-6 seconds |

### TPOT (Time Per Output Token) — Example

**Decode Phase — Sequential Nature:**

```
Token 1 ──► Token 2 ──► Token 3 ──► ... ──► Token N
   │           │           │                   │
  ~10ms       ~10ms       ~10ms               ~10ms

แต่ละ token ต้อง:
1. Attend to ALL previous tokens (KV cache access)
2. Compute new KV entries
3. Update KV cache
4. Sample next token
```

**TPOT ตัวอย่าง (RTX 4090, Llama 3.2 8B Q4_K_M):**

| Batch Size | TPOT (ms/token) | Tokens/sec |
|------------|-----------------|------------|
| 1 | 8-12 | 80-120 |
| 4 | 15-25 | 40-65 |
| 8 | 30-50 | 20-33 |
| 16 | 60-100 | 10-16 |

## Batch Size — Deep Dive

### Batch Size คืออะไร

```
Batch Size = จำนวน sequences ที่ประมวลผลพร้อมกัน

Batch 1:  [Seq A] → process → output A
Batch 4:  [Seq A] ─┐
          [Seq B] ─┼─→ process together → outputs A,B,C,D
          [Seq C] ─┤
          [Seq D] ─┘
```

### Dynamic Batching (Continuous Batching)

```
Problem with static batching:
  - Wait for all sequences in batch to finish
  - Short responses block long responses
  - Inefficient

Continuous Batching (Iteration-level scheduling):
  - Add new requests to batch as slots free up
  - No waiting for "batch" to fill
  - Maximizes GPU utilization
  
  Time ─────────────────────────────────────────────────────►
  
  Step 1: [Req A_____________] [Req B__] [Req C__________]
  Step 2: [Req A_____________] [Req B__] [Req D__] [Req C__________]
  Step 3: [Req A_____________] [Req D__] [Req E_____] [Req C__________]
  
  When Req B finishes → immediately slot in Req D, E, etc.
```

### Optimal Batch Size Recommendations

| Scenario | Batch Size | TPOT | Use Case |
|----------|------------|------|----------|
| Interactive Chat | 1-4 | 10-20ms | Real-time, low latency |
| Document Processing | 4-16 | 30-60ms | Summarization, RAG |
| Batch Processing | 32-64 | 100-200ms | Offline, maximize throughput |
| Very Long Context | 1-2 | 50-100ms | 32K+ tokens |

## Concurrent Users — Multi-User Architecture

### Concurrency vs Parallelism

```
Single user (serial):
  User A ──► [generate 10s] ──► User A ──► [generate 10s] ...
  Total: 30s for 3 users (sequential)

Concurrent (parallel):
  User A ──► [────10s────] ──►
  User B ──► [────10s────] ──►    Total: 10s for 3 users
  User C ──► [────10s────] ──►
  
Same GPU, different time slices
```

### How Concurrency Affects Performance

```
User Count    Batch Size    TPOT (ms)    TTFT (ms)    Throughput
──────────────────────────────────────────────────────────────────
1            1             10           200          100 tok/s
5            5             25           250          200 tok/s
10           10            50           350          200 tok/s
20           20            100          500          200 tok/s
──────────────────────────────────────────────────────────────────

Observation:
  - TPOT increases linearly with users
  - Throughput stays relatively constant
  - TTFT increases slowly (queue wait + prefill)
```

### User Capacity by Hardware

| Hardware | Concurrent Users | Strategy |
|----------|-----------------|----------|
| RTX 4090 24GB | 3-5 | Continuous batching |
| RTX 3090 24GB | 3-5 | Continuous batching |
| A100 40GB | 8-15 | Continuous batching + longer context |
| A100 80GB | 15-30 | Large batches, 128K context |
| Multi-GPU (4×4090) | 15-25 | Tensor parallel |
| Multi-GPU (8×A100) | 50-100+ | Tensor parallel + multi-instance |

## Offloading to CPU/RAM or NVMe

### Why Offload?

```
Problem:
  Model 70B Q4 = ~40 GB
  GPU VRAM: 24 GB
  
  → Can't fit model fully in GPU!
  
Solutions:
  1. CPU offload:  Some layers run on CPU (slow)
  2. NVMe offload: SSD acts as extra memory (very slow but more capacity)
  3. Horizontal scaling:  Multiple GPUs (expensive)
```

### Layer Offloading Strategy

```
Full GPU:     [GPU: 32 layers]           → Fastest
Partial CPU:  [GPU: 24] + [CPU: 8]       → Slower
Heavy CPU:    [GPU: 8]  + [CPU: 24]      → Slow
NVMe hybrid:  [GPU: 8]  + [RAM: 16] + [NVMe: 8] → Very slow
```

### Offloading Performance Comparison

| Strategy | Speed (tok/s) | Latency (ms/tok) | Memory Capacity |
|----------|--------------|------------------|-----------------|
| Full GPU (24GB) | 80-120 | 8-12 | 24GB VRAM |
| GPU + CPU (RAM) | 15-30 | 30-60 | 64GB+ system |
| CPU only (fast) | 5-10 | 100-200 | Unlimited |
| CPU only (slow) | 1-3 | 300-1000 | Unlimited |
| NVMe offload | 0.5-1 | 1000-2000 | Terabytes |

## Why Local LLM is Inevitable

### 1. Cloud AI ราคาจะพุ่งสูง
ปัจจุบัน OpenAI, Anthropic แบกต้นทุนให้เราเพื่อเก็บฐานผู้ใช้ แต่หลังโปรโมชันหมด ราคาจะพุ่งสูง

### 2. Version Control ของตัวเอง
Cloud อัปเดตเมื่อไหร่ก็ไม่สนใจเรา — Local ถ้าจูนจนเจอ workflow ที่ใช่ ก็ freeze ใช้ไปยาวๆ ได้

### 3. Privacy & No Rate Limiting
ข้อมูลไม่ต้องส่งไป server ใคร ใช้ได้ 24/7 ไม่มีลิมิต

---

# Bonus: DGX Spark & UMA Deep Dive

> จากประสบการณ์จริง — DGX Spark vs Mac Studio

## Why Unified Memory Architecture (UMA) Matters

ก่อน UMA การรัน Local LLM ต้องพึ่ง GPU แยกจาก RAM ระบบ ทำให้ต้อง **Offload** ข้อมูลไปมาระหว่าง VRAM และ RAM — เกิดคอขวด (bottleneck) ที่ Memory Bandwidth

### Traditional (Discrete GPU)

```
 CPU                         GPU
(RAM) ◄─────────────────────►(VRAM)
 128GB     bottleneck         24GB
 200GB/s    32GB/s            1000GB/s

If model > VRAM: Offload to RAM = SLOW!
```

### Unified Memory Architecture (UMA)

```
┌──────────────────────────────┐
│         SYSTEM                │
│  ┌────────────────────────┐   │
│  │   CPU + GPU Cores     │   │
│  └────────────────────────┘   │
│            │                   │
│  ┌────────────────────────┐   │
│  │    UNIFIED MEMORY      │   │
│  │    128GB - 512GB       │   │
│  │    No offloading!      │   │
│  └────────────────────────┘   │
└──────────────────────────────┘

✅ No bottleneck between compute and memory
✅ Can address massive memory spaces
✅ Simpler software stack
```

## Three UMA Contenders

### Apple M Series 🍎

**จุดเด่น:**
- Memory bandwidth สูงมาก (~800GB/s)
- Power efficient ไม่ร้อน
- RAM สูงสุด 512GB
- ใช้งานทั่วไปได้ด้วย

**จุดด้อย:**
- พลังประมวลผลต่ำกว่า DGX Spark
- ไม่มี NVFP4
- Limited CUDA ecosystem

### NVIDIA DGX Spark ⚡

**จุดเด่น:**
- 1,000+ TOPS พลังสูงมาก
- **NVFP4 Support!**
- Fine-tune ได้ในเครื่อง
- Scalable: ต่อ 2 เครื่องได้

**จุดด้อย:**
- RAM จำกัด 128GB (ต่อเครื่อง)
- ร้อนมาก (85°C ขึ้นไปง่าย)
- ต้องใช้ Terminal/SSH

### AMD Strix Halo 🔴

**จุดเด่น:**
- ราคาถูกที่สุด
- UMA architecture
- หา Notebook ได้ง่าย

**จุดด้อย:**
- Memory bandwidth ต่ำสุด
- พลังต่ำที่สุด
- ไม่มี NVFP4

> **AMD Strix Halo:** ข้ามไปได้เลยถ้าต้องการพลังสูง เพราะไม่ว่าจะเทียบด้านไหนก็สู้ 2 ตัวบนไม่ได้

## DGX Spark vs Mac Studio: Head-to-Head

| Spec | DGX Spark | Mac Studio (M4 Ultra) | Winner |
|------|-----------|---------------------|--------|
| **Memory Bandwidth** | 273 GB/s | 800+ GB/s | 🏆 Mac |
| **Compute (TOPS)** | 1,000+ | ~32 | 🏆 DGX Spark |
| **Max RAM** | 128GB (ต่อเครื่อง)<br>256GB+ (ต่อกัน 2 เครื่อง) | 512GB | 🏆 Mac (但แพงมาก) |
| **NVFP4 Support** | ✅ มี | ❌ ไม่มี | 🏆 DGX Spark |
| **Ease of Use** | ต้อง SSH / Terminal | Plug and Play | 🏆 Mac |
| **Fine-tune ในเครื่อง** | ✅ ทำได้เลย | ⚠️ ทำได้แต่ช้า | 🏆 DGX Spark |
| **Image/Video Generation** | ✅ CUDA optimized | ⚠️ จำกัดกว่า | 🏆 DGX Spark |
| **Scalability** | 🔗 ต่อ 2 เครื่องได้<br>สูงสุด 256GB | ❌ ต่อไม่ได้ | 🏆 DGX Spark |

## Pre-fill vs Token Generation: ไม่ใช่แค่ "เร็วกว่า"

LLM inference แบ่งเป็น 2 ช่วงหลัก ซึ่งแต่ละช่วงต้องการทรัพยากรต่างกัน:

### PHASE 1: PRE-FILL (Prompt Processing)
- Input: "โค้ดทั้งหมด + Context + คำถามของผม"
- Process: ประมวลผล Prompt ทั้งหมดก่อนเริ่มตอบ
- Output: Time to First Token (TTFT)
- **🏆 DGX Spark ชนะ!** ประมวลผลได้เร็วกว่าเห็นๆ
- ใช้: **COMPUTE POWER** เป็นหลัก (TOPS)

### PHASE 2: TOKEN GENERATION (Decoding)
- Input: เริ่มรับ token ทีละตัวจนครบ
- Process: ทำนาย token ถัดไป → พิมพ์ → ทำนายต่อ
- Output: Tokens per Second (TPS)
- **🏆 Mac Studio ตีตื้นกลับมา!** พิมพ์เร็วกว่า
- ใช้: **MEMORY BANDWIDTH** เป็นหลัก (GB/s)

### ความหมายในการเลือกใช้งานจริง

| Use Case | Heavy ช่วงไหน? | แนะนำ |
|----------|---------------|--------|
| Code ที่มี Context ยาว (Few-shot + CoT) | Pre-fill (DGX Spark ได้เปรียบ) | 🏆 DGX Spark |
| แปล Codebase ใหญ่ๆ | Pre-fill (DGX Spark ได้เปรียบ) | 🏆 DGX Spark |
| General chat สั้นๆ | Token Gen (Mac ได้เปรียบ) | 🏆 Mac |
| งานเอกสาร สรุปบทความ | ขึ้นกับ Context length | Context ยาว = DGX Spark |
| RAG (Retrieval Augmented Gen) | Pre-fill (ดึง context เยอะ) | 🏆 DGX Spark |
| Interactive coding (ให้ตอบทีละบรรทัด) | Token Gen (Mac ได้เปรียบ) | 🏆 Mac |

> **Pro Tip:** EXO Lab ทดสอบแล้วว่าเอา DGX Spark ทำ Pre-fill แล้วส่งต่อให้ Mac Studio ทำ Token Generation จะได้ความเร็วสูงสุด แต่ต้องมีทุน 2 เครื่องก่อน 😅

## NVFP4: พระเอกของ DGX Spark

นี่คือเหตุผลหลักที่เลือก DGX Spark และเป็นสิ่งที่ค่ายอื่น **ไม่มี!**

### หลักการ
NVIDIA ทำให้ DGX Spark รัน 4-bit quantization โดยรักษาความแม่นยำได้เกือบเท่า FP16!

### เปรียบเทียบความแม่นยำ (Accuracy)

| Precision | ความแม่นยำ |
|-----------|-----------|
| FP16 (Baseline) | 100% |
| FP8 (8-bit) | ~99.9% |
| FP4 (4-bit ธรรมดา) | ~92-97% |
| **NVFP4 (4-bit NVIDIA)** | **~99.5%** |

> 💡 **NVFP4 แม่นกว่า FP4 ธรรมดา ถึง ~4.5% เลยทีเดียว! แถมใกล้เคียง FP8 มาก!**

### NVFP4 ทำงานยังไง?

เหมือนเราได้ Zip ไฟล์มาแล้วใช้งานได้เลย ไม่ต้อง Unzip ก่อน! มันถูกออกแบบมาให้รัน Native บน Hardware ระดับต่ำ

| Precision | ความแม่นยำ | DGX Spark | Mac Studio |
|-----------|-----------|------------|------------|
| FP16 (native) | 100% | ✅ | ✅ |
| FP8 / INT8 | ~99.9% | ✅ Native | ✅ (แต่ไม่ Native) |
| FP4 ธรรมดา | ~92-97% | ✅ | ⚠️ ต้องใช้ FP8 แทน |
| **NVFP4** | **~99.5%** | ✅ **Native** | ❌ ไม่มี |

> **ข้อจำกัด:** NVFP4 ต้องการโมเดลที่ถูก Quantize เป็น NVFP4 format ซึ่งยังไม่แพร่หลายเท่า GGUF Q4_K_M แต่ community ก็เริ่มมีโมเดล NVFP4 เพิ่มขึ้นเรื่อยๆ

## DGX Spark: ข้อดี ข้อด้อย ที่ต้องรู้

### ✅ ข้อดี
- **พลังประมวลผลเหนือชั้น** — Supercomputer ขนาดเล็กที่ราคาจับต้องได้
- **NVFP4 Support** — ได้ความแม่นยำสูงสุดที่ 4-bit
- **Fine-tune ได้ในเครื่อง** — เคยต้องเช่า Colab ตอนนี้ไม่ต้องแล้ว!
- **Scalable** — ซื้อ 2 เครื่องมาต่อกันได้เลย (128+128=256GB)
- **DGX OS + CUDA AI Stack** — เซ็ตอัพง่าย มี software stack พร้อม
- **Image/Video Generation** — CUDA optimized ทำให้เร็วกว่า Mac Studio มาก

### ❌ ข้อด้อย
- **ต้องใช้ Terminal/SSH** — ใครไม่ถนัด Command line ไม่แนะนำ
- **ร้อนมาก!** — อุณหภูมิขึ้น 85°C ได้ง่าย ต้องดูแล Airflow
- **RAM จำกัด 128GB** — รันโมเดลใหญ่มากๆ ไม่ได้ตัวเดียว (ต้องต่อ 2 เครื่อง)
- **ต้องปรับตั้งค่าเยอะ** — Kernel panic, OOM, Thermal issues ต้องรู้วิธีแก้
- **Memory bandwidth ต่ำกว่า Mac** — Token generation ช้ากว่า

> **ประสบการณ์จริง:** ช่วงแรก Kernel panic Hard Shutdown ไป 3-4 รอบ (OOM 3 ครั้ง + Thermal 1 ครั้ง) — ต้องศึกษา log และปรับแต่งเยอะกว่าจะลงตัว

## DGX Spark vs Mac Studio: งบเท่าไหร่ดี?

| สถานการณ์ | แนะนำ | เหตุผล |
|-----------|-------|--------|
| งบจำกัด แต่อยากได้พลัง | DGX Spark | ได้ TOPS สูงสุดในราคา |
| ไม่ถนัด Terminal | Mac Studio | ใช้ได้เลย ไม่ต้องตั้งค่าอะไร |
| ต้องการ RAM 128GB+ ราคาถูก | DGX Spark × 2 ต่อกัน | 256GB ในราคาประหยัดกว่า Mac Studio 512GB |
| ต้องการ RAM 512GB+ | Mac Studio 512GB | DGX Spark ทำไม่ได้ (ต้อง 4 เครื่อง = แพงมาก) |
| ใช้งานทั่วไป + เขียนโค้ดเป็นหลัก | Mac Studio M4 Max 256GB | เพียงพอ + ใช้ง่าย + token gen เร็ว |
| Fine-tune + Image/Video Gen | DGX Spark | CUDA ecosystem + พลังสูง |
| Heavy Prompt + Long Context | DGX Spark | Pre-fill เร็วกว่าเยอะ |

> **Mac Mini / Mac Studio Max:** ข้ามเลยถ้าไปสาย Mac แล้วต้องจัด Studio ตัว Ultra RAM 256GB ขึ้นไปเท่านั้น! ตัว Max RAM 64GB ทำอะไรได้ไม่เต็มศักยภาพ

## Quantization: ยัดโมเดลใหญ่ลง RAM น้อย

ถ้าอยากรันโมเดลใหญ่อย่าง **Qwen3.5 122B** (ขนาด ~245GB FP16) บน RAM 128GB ต้อง Quantize:

| Format | ขนาด (122B model) | DGX Spark (128GB) | Mac Studio 256GB | ความแม่นยำ |
|--------|------------------|-------------------|-----------------|------------|
| FP16 | ~245 GB | ❌ | ❌ | 100% |
| FP8 / INT8 | ~132 GB | ⚠️ รัน 1 เครื่องไม่ได้ | ✅ | ~99.9% |
| FP4 ธรรมดา | ~80 GB | ✅ | ✅ | ~92-95% |
| **NVFP4** | ~80 GB | ✅ **Native** | ❌ ไม่มี | **~99.5%** |

> **เปรียบเทียบ:** NVFP4 บน DGX Spark (1 เครื่อง) = ความแม่นยำ ~99.5% ที่ 80GB
> FP8 บน Mac Studio 256GB = ความแม่นยำ ~99.9% ที่ 132GB
>
> **DGX Spark ใช้ RAM น้อยกว่า 40% แต่แม่นยำใกล้เคียง!**

## DGX Spark Variants: MSI vs NVIDIA Founder Edition

| แง่ | NVIDIA Founder | MSI |
|-----|---------------|-----|
| สถาปัตยกรรม | เหมือนกัน | เหมือนกัน |
| Vapor Chamber | ❌ ธรรมดา | ✅ ครอบคลุมกว่า |
| Housing/ Cooling | ร้อนกว่า 5-10% | ดีกว่า |
| ราคา | อาจถูกกว่า | อาจแพงกว่าเล็กน้อย |
| แนะนำ | ถ้าราคาต่างกันมาก | ถ้าซื้อใหม่ |

> **รีวิวจริง:** อุณหภูมิ MSI ต่ำกว่า NVIDIA Founder Edition ราวๆ 5-10°C ซึ่งทำให้ performance ดีกว่าและเครื่องทนทานกว่าในระยะยาว

---

# Complete Formulas Reference

```
┌─────────────────────────────────────────────────────────────┐
│                    KEY FORMULAS                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. WEIGHT SIZE                                            │
│     weight_size = params × bytes_per_param                  │
│     Example: 7B × 0.5 bytes (Q4) = 3.5 GB                  │
│                                                             │
│  2. KV CACHE SIZE                                         │
│     kv_size = 2 × n_layers × n_kv_heads                    │
│              × seq_len × head_dim × bytes                   │
│     Example: 2×32×8×4096×128×2 = 512 MB (GQA, FP16)       │
│                                                             │
│  3. ATTENTION FLOP (prefill)                              │
│     flops = 2 × n_tokens × n_layers × hidden_dim × seq      │
│             (approximation)                                 │
│                                                             │
│  4. ATTENTION FLOP (decode per token)                     │
│     flops = 2 × n_layers × hidden_dim × (2 × total_ctx)     │
│             (attention scales with total context)           │
│                                                             │
│  5. GPU LAYER FIT                                         │
│     layers_gpu = (vram - kv_overhead) / (model_size/layers  │
│                 + kv_per_layer + activations_per_layer)     │
│                                                             │
│  6. THROUGHPUT (tokens/second)                            │
│     throughput = 1 / TPOT                                  │
│     Or: batch_size × tokens / total_time                    │
│                                                             │
│  7. TTFT ESTIMATE                                         │
│     ttft = prefill_time + queue_wait                        │
│     prefill ≈ n_prompt_tokens × layers × flops_per_token    │
│                                                             │
│  8. MEMORY FOR SEQUENCE (total)                           │
│     total = model_weights + kv_cache + activations          │
│           + output_buffer + overhead                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

# Quick Reference

## Memory Requirements Summary (7B Model)

| Format | Size |
|--------|------|
| FP16 | ~14 GB |
| Q8_0 | ~7 GB |
| Q6_K | ~5.5 GB |
| Q5_K_M | ~4.9 GB |
| **Q4_K_M** | **~4.2 GB** |

## Popular Models 2025-2026

**Best for Coding:**
- Qwen-2.5-Coder-7B-Instruct
- Deepseek-Coder-6.7B-Instruct
- Codestral-22B
- Llama-3.1-70B-Instruct

**Best for Chat/General:**
- Llama-3.1-8B-Instruct
- Mistral-7B-Instruct-v0.3
- Qwen-2.5-7B-Instruct
- Phi-4 14B

**Best for Long Context:**
- Llama-3.1-8B-Instruct (128K)
- Command R7B (128K)
- Qwen-2.5-72B-Instruct (128K)

**Best for Multimodal:**
- Llama-3.2-90B-Vision
- Gemma-3-27B-IT

---

## Next Steps

**Topics for future updates:**
- Fine-tuning locally with Axolotl / LoRA
- Deployment with Docker + Kubernetes
- Monitoring with Prometheus + Grafana
- Security hardening
- Cost optimization

---

> **Document Version:** 4.1  
> **Last Updated:** March 2026
