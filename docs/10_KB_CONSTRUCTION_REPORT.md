# Báo Cáo Xây Dựng Psychology Knowledge Base Cho RAG-XPR

## 1. Executive Summary

Knowledge Base hiện tại là `psych_kb_ocean_v3`, được thiết kế để phục vụ RAG-XPR trong bài toán
personality detection có giải thích được. Mục tiêu chính của KB không phải là nhồi thật nhiều text
tâm lý học vào vector database, mà là tạo một lớp tri thức có cấu trúc giúp model đi theo chuỗi:

```text
utterance / evidence -> psychological state -> OCEAN trait polarity -> final prediction
```

KB hiện tại ưu tiên OCEAN vì dataset chính để đánh giá explainability là PersonalityEvd. Các nguồn
MBTI vẫn được giữ để hỗ trợ benchmark và generalization, nhưng không phải trọng tâm khoa học của
phiên bản KB này.

Trạng thái hiện tại:

| Thành phần | Giá trị |
|------------|---------|
| KB version | `psych_kb_ocean_v3` |
| Qdrant collection | `psych_kb_ocean_v3` |
| Embedding model | `BAAI/bge-base-en-v1.5` |
| Vector size | `768` |
| Chunk count | `1007` |
| Embedding shape | `(1007, 768)` |
| Config hash | `0f0917c3bcc6a72f11751516c4dab8f362fdf18ff3a456419bbe58098632bab5` |
| Chunk hash | `1704e4fb83b2c3bb94f345a78a56c7aee3312e934e53d58ee4a06292d4578d09` |

Các kiểm tra gần nhất:

| Kiểm tra | Kết quả |
|----------|---------|
| KB audit | `1007` chunks, `0` invalid, `0` duplicate |
| Held-out exact leakage | `0` matches với `data/processed/personality_evd/test.jsonl` |
| BM25 retrieval Recall@5 | `0.840` |
| BM25 retrieval MRR | `0.511` |
| Min per-trait Recall@5 | `0.800` |
| Qdrant verify | `1007` points indexed |

## 2. Vì Sao Cần KB Riêng Cho RAG-XPR

Các dataset hiện tại không cung cấp một Knowledge Base tâm lý học đi kèm. Điều này tạo ra một vấn
đề lớn cho nghiên cứu explainability:

- Nếu không có KB, phần giải thích dễ trở thành “LLM tự biết” hoặc “LLM tự diễn giải”.
- Nếu chỉ dùng label dataset, model có thể học shortcut giữa câu và nhãn nhưng không có grounding
  vào khái niệm tâm lý học.
- Nếu dùng raw textbook/manual dài, retrieval dễ nhiễu, khó kiểm soát bản quyền, khó tái lập, và
  không map trực tiếp sang state/trait cần cho PersonalityEvd.
- Nếu dùng few-shot quá nhiều trong cùng collection với định nghĩa, retrieval dễ bị example
  pollution: câu hỏi về state lại retrieve một prompt example dài thay vì retrieve tri thức gốc.

Do đó KB hiện tại được xây theo hướng hybrid pragmatic:

- Dùng tri thức tâm lý học citable/paraphrased làm backbone.
- Dùng annotation train/valid của PersonalityEvd English làm evidence mapping thực nghiệm.
- Dùng rules ngắn cho abstention và state-trait aggregation để hỗ trợ explainability.
- Dùng metadata mạnh để filter/rerank theo framework, category, trait, pole, source, quality.
- Không đưa PersonalityEvd test content vào KB.

## 3. Design Principles

### 3.1 OCEAN-first

OCEAN là trọng tâm vì PersonalityEvd đánh giá theo Big Five và là dataset duy nhất trong project có
evidence/state annotations phục vụ đo explainability. MBTI, Pandora, Essays vẫn hữu ích để chứng
minh generalization, nhưng không có ground-truth evidence/state tương đương.

Tác động của quyết định này:

- KB có nhiều state, facet, marker, linguistic correlate cho OCEAN hơn MBTI.
- Retrieval Step 2 và Step 3 trong CoPE ưu tiên category phù hợp với OCEAN reasoning.
- PersonalityEvd trở thành benchmark chính để đánh giá đóng góp khoa học của RAG-XPR.

### 3.2 English-only KB

KB dùng tiếng Anh vì pipeline hiện tại dùng English prompts, English embedding model, và source
enrichment được tạo từ `data/raw/personality_evd_en`.

Lý do chọn English:

- Giảm mismatch giữa query, document, prompt và embedding model.
- Tránh nhiễu do mixed Chinese-English raw text trong PersonalityEvd gốc.
- Dễ reuse citation từ literature tiếng Anh.
- Dễ debug trực quan qua JSONL, dashboard, và retrieved chunks.

### 3.3 Citable paraphrase, không copy manual dài

Các manual/textbook/paper được đưa vào KB dưới dạng paraphrase ngắn và metadata citation. Repo không
lưu các đoạn copyrighted manual dài.

Ý nghĩa:

- Giảm rủi ro bản quyền.
- Giữ mỗi chunk như một knowledge atom nhỏ, dễ retrieve.
- Khi viết paper, ta có thể cite source trong metadata thay vì trích đoạn dài từ KB.

### 3.4 Record-aware chunking

KB không dùng chunking generic kiểu `512/64` cho mọi source. Phần lớn record đã ngắn và có nghĩa
độc lập, nên được giữ dạng atomic.

Chiến lược hiện tại:

| Category | Chunking |
|----------|----------|
| `trait_definition` | atomic |
| `facet_definition` | atomic |
| `state_definition` | atomic |
| `behavioral_marker` | atomic |
| `linguistic_correlate` | atomic |
| `type_description` | atomic |
| `cognitive_function` | atomic |
| `evidence_mapping_example` | atomic, `max_tokens=320` |
| `few_shot_example` | structured blocks: `INPUT+STEP1`, `STEP2`, `STEP3` |

Lý do:

- Definition/marker/state là knowledge atoms, tách nhỏ hơn sẽ làm mất nghĩa.
- Evidence mapping cần giữ quote, label, reasoning trong cùng record nên cần ngưỡng lớn hơn.
- Few-shot example dài nên tách theo bước reasoning để không lấn át retrieval.

### 3.5 Embed text khác human text

Mỗi chunk có hai field:

| Field | Vai trò |
|-------|---------|
| `text` | Nội dung đầy đủ để người đọc và LLM sử dụng |
| `embed_text` | Text có semantic anchor để embed và retrieve tốt hơn |

Ví dụ:

```text
Evidence mapping. Framework: OCEAN. Trait: C HIGH. Source: personality_evd_trainval_evidence.
Evidence mapping from PersonalityEvd train/valid. Evidence quote: ...
```

Ý nghĩa:

- Dense retriever nhìn thấy rõ framework, trait, pole, source.
- Các query ngắn như “planned schedule carefully” dễ map sang `C HIGH`.
- Metadata và semantic anchor cùng hỗ trợ filtering, reranking và debug.

## 4. Schema Hiện Tại

Mỗi source record dùng JSONL schema chuẩn:

```json
{
  "chunk_id": "personality_evd_evidence_mapping_00001",
  "text": "Human-readable KB text.",
  "metadata": {
    "framework": "ocean",
    "category": "evidence_mapping_example",
    "source_id": "personality_evd_trainval_evidence",
    "citation": "Sun et al., 2024; PersonalityEvd train/valid evidence annotations",
    "source_type": "dataset_annotation",
    "quality_tier": "A",
    "license_status": "project_seed",
    "trait": "C",
    "pole": "HIGH",
    "language": "en",
    "split_safety": "train_val_only_no_test_content"
  }
}
```

Các metadata quan trọng:

| Field | Ý nghĩa |
|-------|---------|
| `framework` | `ocean`, `mbti`, hoặc `both` |
| `category` | Loại tri thức, dùng cho chunking và retrieval filter |
| `source_id` | Source registry id, dùng để trace provenance |
| `quality_tier` | Độ tin cậy nguồn: `A`, `B`, `C` |
| `trait` | OCEAN trait hoặc MBTI dimension/type signal nếu có |
| `pole` | `HIGH`, `LOW`, `BOTH`, hoặc missing nếu không áp dụng |
| `state_label` | Tên psychological state nếu category là state |
| `mapping_type` | Loại evidence mapping/rule |
| `split_safety` | Ghi rõ record có an toàn với test split hay không |

Quality tier:

| Tier | Ý nghĩa | Cách dùng |
|------|---------|----------|
| `A` | Paper/manual/dataset annotation có provenance rõ | Có thể dùng trong paper |
| `B` | Nguồn học thuật thứ cấp hoặc public documentation đáng tin | Dùng được nhưng cần flag |
| `C` | Legacy seed, synthetic/project-authored hoặc nguồn cần audit thêm | Dùng để hỗ trợ nhưng không claim như source học thuật chính |

## 5. Nguồn Dữ Liệu Trong KB Hiện Tại

### 5.1 Source files

| Source file | Records | Vai trò |
|-------------|---------|--------|
| `behavioral_markers.jsonl` | `215` | Behavioral cues cho OCEAN/MBTI |
| `bfi2_item_anchors.jsonl` | `30` | BFI-2 paraphrased anchors theo trait/facet/pole |
| `linguistic_correlates.jsonl` | `118` | Word/language correlates từ literature |
| `psychological_states.jsonl` | `98` | State labels map sang trait signals |
| `ocean_traits.jsonl` | `15` | OCEAN domain definitions |
| `ocean_curated_v1.jsonl` | `20` | Curated OCEAN definitions/markers |
| `ocean_facets.jsonl` | `60` | Facet definitions cho OCEAN |
| `personality_evd_evidence_mappings.jsonl` | `224` | Train/valid evidence mappings từ PersonalityEvd English |
| `abstention_and_insufficient_evidence.jsonl` | `5` | Rules cho insufficient evidence/abstention |
| `state_trait_aggregation_rules.jsonl` | `5` | Rules cho state-to-trait aggregation |
| `cope_few_shot.jsonl` | `25` | CoPE examples, split thành reasoning blocks |
| `mbti_types.jsonl` | `96` | MBTI type portraits |
| `mbti_cognitive_functions.jsonl` | `32` | MBTI cognitive functions |

Ngoài ra còn legacy files:

| Source file | Vai trò |
|-------------|--------|
| `mbti_definitions.jsonl` | Minimal MBTI legacy definitions |
| `ocean_definitions.jsonl` | Minimal OCEAN legacy definitions |
| `few_shot_examples.jsonl` | Legacy few-shot example |

### 5.2 Source ids nổi bật

| Source id | Chunks | Ý nghĩa |
|-----------|--------|---------|
| `personality_evd_trainval_evidence` | `224` | Mapping thực tế từ evidence quote sang OCEAN level |
| `mairesse_2007_linguistic_cues` | `76` | Linguistic/personality cues |
| `schwartz_2013_social_media` | `73` | Language correlates từ social media |
| `yarkoni_2010_100k_words` | `71` | Word-personality correlates |
| `neo_pi_r_costa_mccrae` | `69` | Big Five/facet backbone |
| `pennebaker_king_1999` | `64` | LIWC/personality language relations |
| `bfi2_paraphrased_item_anchors` | `30` | Short BFI-2-inspired behavioral anchors |
| `whole_trait_theory_fleeson_2015` | `10` | Abstention và state-trait aggregation rules |

### 5.3 Category distribution

| Category | Chunks | Ý nghĩa |
|----------|--------|---------|
| `behavioral_marker` | `250` | Hành vi observable map sang trait/pole |
| `evidence_mapping_example` | `234` | Dataset-grounded mapping và rules |
| `linguistic_correlate` | `123` | Tín hiệu ngôn ngữ liên quan đến trait |
| `type_description` | `104` | MBTI type descriptions |
| `state_definition` | `103` | Psychological states cho Step 2 |
| `few_shot_example` | `76` | CoPE reasoning examples đã split |
| `facet_definition` | `60` | OCEAN facets |
| `cognitive_function` | `32` | MBTI functions |
| `trait_definition` | `25` | OCEAN/legacy trait definitions |

### 5.4 Framework distribution

| Framework | Chunks | Diễn giải |
|-----------|--------|-----------|
| `ocean` | `528` | Trọng tâm cho PersonalityEvd và explainability |
| `both` | `295` | Tri thức dùng được cho cả OCEAN/MBTI |
| `mbti` | `184` | Hỗ trợ benchmark MBTI và generalization |

### 5.5 Quality distribution

| Quality tier | Chunks | Diễn giải |
|--------------|--------|-----------|
| `A` | `737` | Nguồn chính, có provenance/citation rõ |
| `B` | `130` | Nguồn phụ đáng tin |
| `C` | `140` | Legacy/synthetic/project seed, cần xem như hỗ trợ |

## 6. PersonalityEvd Enrichment Được Tạo Như Nào

Script:

```text
scripts/build_kb_enrichment_sources.py
```

Input canonical:

```text
data/raw/personality_evd_en/Dataset/dialogue.json
data/raw/personality_evd_en/Dataset/EPR-State Task/train_annotation.json
data/raw/personality_evd_en/Dataset/EPR-State Task/valid_annotation.json
```

Không dùng làm KB source:

```text
data/raw/personality_evd_en/Dataset/EPR-State Task/test_annotation.json
```

Quy trình:

1. Đọc dialogue English và annotation train/valid.
2. Parse turn bằng regex dạng `Utterance <id> <speaker> said: <utterance>`.
3. Map trait name sang OCEAN code: `openness -> O`, `conscientiousness -> C`,
   `extraversion -> E`, `agreeableness -> A`, `neuroticism -> N`.
4. Map level sang `HIGH`, `LOW`, hoặc `UNKNOWN`.
5. Reconstruct quote từ `utt_id` và target speaker.
6. Chỉ sinh positive mapping nếu quote reconstruct được.
7. Sinh UNKNOWN mapping theo abstention semantics, không đưa raw test explanation vào KB.
8. Sampling deterministic seed `42`.
9. Giới hạn mặc định `20` HIGH/LOW examples mỗi trait-level và `10` UNKNOWN examples mỗi trait.
10. Lọc exact-string leakage bằng `data/processed/personality_evd/test.jsonl`.

Kết quả gần nhất:

| Output | Records |
|--------|---------|
| Raw sampled candidates | `250` |
| Excluded by exact held-out leakage filter | `26` |
| Final `personality_evd_evidence_mappings.jsonl` | `224` |

Việc dùng held-out processed test trong script chỉ có một mục đích: loại bỏ exact string overlap khỏi
KB. Script không lấy test labels, không lấy test explanations, không lấy test examples để sinh tri
thức.

## 7. Vì Sao Chọn Các Nguồn Này

### 7.1 OCEAN trait/facet definitions

Nguồn như Costa & McCrae, BFI-2/Soto & John, Big Five literature được chọn vì chúng cung cấp
backbone tâm lý học cho domains và facets. Đây là lớp tri thức trả lời câu hỏi “trait này nghĩa là
gì?” và “facet nào cấu thành trait này?”.

Nếu thiếu lớp này:

- Model có thể map evidence sang label nhưng không giải thích được trait theo khái niệm chuẩn.
- Step 3 dễ trở thành label generation thay vì trait inference.
- Paper khó claim rằng explanation có grounding tâm lý học.

### 7.2 Psychological states

PersonalityEvd đánh giá evidence/state, nên KB cần state layer. State definitions giúp Step 2 map
utterance cụ thể sang trạng thái ngắn hạn, ví dụ social withdrawal, planning orientation,
emotional volatility, altruistic activation.

Nếu thiếu state layer:

- Model phải nhảy thẳng từ quote sang trait.
- Explanation chain ngắn và khó kiểm chứng.
- Evidence F1 hoặc state consistency khó cải thiện.

### 7.3 Behavioral markers

Behavioral markers mô tả hành vi observable, ví dụ planning carefully, seeking solitude, helping
others, worrying repeatedly. Chúng là cầu nối giữa text và trait.

Lý do chọn:

- Text trong datasets thường là hành vi, lời nói, reaction, hoặc preference.
- Marker ngắn giúp retrieval trả về tri thức đúng sát câu evidence.
- Marker có trait/pole metadata nên dễ filter và debug.

### 7.4 Linguistic correlates

Các paper về language-personality như Mairesse, Yarkoni, Schwartz, Pennebaker/LIWC được dùng để bổ
sung tín hiệu khi evidence không nói thẳng hành vi nhưng thể hiện qua language use.

Vai trò:

- Hỗ trợ datasets long-form như Essays.
- Hỗ trợ MBTI/Pandora/Pandora-style text khi evidence là style hoặc word usage.
- Cung cấp weak signals, không nên dùng như bằng chứng tuyệt đối.

### 7.5 PersonalityEvd train/valid evidence mappings

Đây là phần enrichment thực dụng nhất cho RAG-XPR. Nó cho model ví dụ thật về cách annotation của
task map từ utterance sang trait level.

Lý do dùng train/valid:

- Train/valid là nguồn hợp lệ để xây prompting/retrieval support.
- Examples bám sát task distribution của PersonalityEvd.
- Giúp RAG-XPR học “format of reasoning” mà không chạm vào test.

Lý do không dùng test:

- Test phải giữ sạch để đánh giá scientific contribution.
- Nếu test quote vào KB, retrieval có thể trả về chính sample cần dự đoán.
- Khi đó performance/explanation không còn đáng tin.

### 7.6 Abstention và insufficient evidence rules

PersonalityEvd có nhiều trường hợp `cannot be determined`. Nếu KB chỉ chứa positive examples, model
có xu hướng forced-choice HIGH/LOW ngay cả khi evidence yếu.

Rules hiện tại bao phủ:

- quote rỗng hoặc không reconstruct được.
- single weak cue.
- social politeness không đủ để kết luận Agreeableness.
- situational anxiety không đủ để kết luận stable Neuroticism.
- một dialogue đơn lẻ không đủ cho trait-level conclusion nếu thiếu pattern.

Tác động:

- Giảm overclaiming.
- Tăng khả năng nói “insufficient evidence”.
- Làm explanation trung thực hơn khi evidence yếu.

### 7.7 State-trait aggregation rules

RAG-XPR không chỉ cần state detection mà còn cần aggregate states thành trait. Rules từ Whole Trait
Theory giúp model hiểu rằng trait là distribution của states qua thời gian, không phải một hành vi
đơn lẻ.

Rules hiện tại bao phủ:

- state là biểu hiện ngắn hạn.
- trait cần pattern across states/contexts.
- state-trait disagreement là hợp lệ.
- repeated evidence mạnh hơn isolated evidence.
- conflicting states nên giảm confidence thay vì force conclusion.

Tác động:

- Step 3 có tiêu chuẩn rõ hơn để aggregate.
- Explanation có thể nêu conflict và uncertainty.
- Giảm lỗi suy diễn từ một quote quá nổi bật.

### 7.8 BFI-2 paraphrased anchors

BFI-2 item anchors được paraphrase ngắn theo trait, pole, facet. Chúng không copy manual dài.

Lý do thêm:

- BFI-2 có facet structure rõ, phù hợp Big Five.
- Anchors giúp retrieval có câu ngắn sát hành vi thực tế.
- Facet-level anchors làm KB chi tiết hơn domain-level definitions.

## 8. Build Pipeline

Command sinh enrichment sources:

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/build_kb_enrichment_sources.py
```

Command parse:

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/build_kb.py --step parse --config configs/kb_config.yaml
```

Command embed CPU-safe:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false nice -n 15 ionice -c3 \
uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/build_kb.py --step embed --config configs/kb_config.yaml
```

Command index Qdrant:

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/build_kb.py --step index --config configs/kb_config.yaml
```

Command verify:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false nice -n 15 ionice -c3 \
uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/build_kb.py --step verify --config configs/kb_config.yaml
```

Các artifact chính:

| Artifact | Vai trò |
|----------|---------|
| `data/knowledge_base/sources/*.jsonl` | Source-level KB, dễ review thủ công |
| `data/knowledge_base/chunks.jsonl` | Parsed chunks có `text`, `embed_text`, `metadata` |
| `data/knowledge_base/embeddings.npy` | Dense vectors để index Qdrant |
| `data/knowledge_base/kb_manifest.json` | Hash, config, counts, reproducibility metadata |
| `data/knowledge_base/reports/kb_audit.json` | Schema/provenance/leakage audit |
| `data/knowledge_base/reports/kb_dashboard.html` | Dashboard review trực quan |

## 9. Retrieval Policy Trong RAG-XPR

CoPE pipeline dùng category-filtered retrieval:

Step 2, state identification:

```text
state_definition
behavioral_marker
linguistic_correlate
evidence_mapping_example
```

Step 3, trait inference:

```text
trait_definition
facet_definition
evidence_mapping_example
```

Lý do:

- Step 2 cần map quote sang state/marker/correlate.
- Step 3 cần definitions/facets/rules để aggregate thành trait.
- `evidence_mapping_example` được phép ở cả hai step vì nó chứa cả positive mappings, abstention
  rules và aggregation rules.
- `few_shot_example` không nằm trong default retrieval path để tránh long example pollution.

## 10. Tác Động Đến Model

### 10.1 Tác động lên evidence-to-state grounding

KB cung cấp state definitions và behavioral markers để model không phải tự nghĩ ra state labels. Khi
gặp câu evidence, retriever có thể trả về các chunk như planning, social withdrawal, worry,
helping behavior, curiosity.

Kỳ vọng:

- Step 2 nhất quán hơn.
- State labels ít hallucination hơn.
- Explanation chain có reference rõ hơn.

### 10.2 Tác động lên trait inference

Trait definitions, facets và aggregation rules giúp Step 3 không suy diễn quá nhanh từ một cue đơn
lẻ. Model có thêm tiêu chuẩn để xem evidence là isolated, repeated, conflicting hay insufficient.

Kỳ vọng:

- Classification macro F1 ổn định hơn khi evidence nhiều nhiễu.
- Confidence calibration tốt hơn.
- Explanation phản ánh uncertainty thay vì forced conclusion.

### 10.3 Tác động lên explainability

KB v3 làm explanation có thể audit theo ba lớp:

```text
retrieved chunk -> psychological interpretation -> final trait prediction
```

Điều này quan trọng vì RAG-XPR không chỉ cần đúng label, mà còn cần chứng minh model dùng tri thức
phù hợp.

Kỳ vọng:

- Evidence F1 và grounding có thể đo được trên PersonalityEvd.
- Người review có thể xem chunk source, category, trait, pole.
- Paper có thể report KB hash và collection version để tái lập.

### 10.4 Tác động lên generalization

MBTI, Pandora, Essays không có evidence labels mạnh như PersonalityEvd. KB giúp model dùng chung
framework reasoning thay vì overfit dataset-specific labels.

Kỳ vọng:

- Essays được hỗ trợ bởi OCEAN definitions, facets, long-form language correlates.
- MBTI vẫn có type/function knowledge để benchmark.
- Pandora-style text có thể benefit từ behavioral markers và linguistic correlates.

## 11. Leakage Control

Các rule leakage hiện tại:

- Không đọc `test_annotation.json` để sinh KB source.
- Không đưa PersonalityEvd test explanations vào KB.
- Positive evidence mapping chỉ lấy từ train/valid English raw annotations.
- Script enrichment mặc định dùng `data/processed/personality_evd/test.jsonl` chỉ để lọc exact
  string overlap.
- Audit chạy exact held-out leakage check trên built chunks.

Kết quả hiện tại:

```text
KB audit complete: 1007 chunks, 0 invalid, 0 duplicates
Held-out exact leakage matches: 0
```

Lưu ý quan trọng:

- Exact leakage check không chứng minh không có semantic overlap.
- Nó chỉ đảm bảo không copy nguyên chuỗi held-out dài vào KB.
- Đây là mức kiểm soát cần thiết tối thiểu cho reproducible evaluation.

## 12. Reproducibility

KB v3 có các cơ chế tái lập:

- `kb_version`: `psych_kb_ocean_v3`.
- `collection_name`: `psych_kb_ocean_v3`.
- `config_hash` trong manifest.
- `chunks_hash` trong manifest.
- deterministic sampling seed `42` cho PersonalityEvd enrichment.
- deterministic Qdrant point ids derived from `chunk_id`.
- source files được lưu trong repo dưới dạng JSONL.

Khi chạy experiment, output nên log:

```text
kb_version=psych_kb_ocean_v3
kb_hash=1704e4fb83b2c3bb94f345a78a56c7aee3312e934e53d58ee4a06292d4578d09
qdrant_collection=psych_kb_ocean_v3
embedding_model=BAAI/bge-base-en-v1.5
```

## 13. Cách Review Trực Quan

Mở dashboard:

```text
data/knowledge_base/reports/kb_dashboard.html
```

Đọc source-level dumps:

```text
data/knowledge_base/psychology_kb_source_dump_v1.jsonl
data/knowledge_base/ocean_knowledge_v1.jsonl
data/knowledge_base/cope_examples_v1.jsonl
```

Đọc source files gốc:

```text
data/knowledge_base/sources/personality_evd_evidence_mappings.jsonl
data/knowledge_base/sources/abstention_and_insufficient_evidence.jsonl
data/knowledge_base/sources/state_trait_aggregation_rules.jsonl
data/knowledge_base/sources/bfi2_item_anchors.jsonl
```

Chạy audit:

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/audit_kb.py
```

Chạy retrieval QA:

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/evaluate_kb_retrieval.py --method bm25
```

## 14. Limitations

KB hiện tại đã đủ tốt cho v3, nhưng vẫn có các giới hạn:

- `quality_tier=C` vẫn còn `140` chunks legacy/synthetic, nên khi viết paper cần ưu tiên cite tier A/B.
- Retrieval QA hiện chỉ có `25` queries, chưa đủ đại diện cho toàn bộ PersonalityEvd.
- Exact leakage check chưa bắt semantic paraphrase overlap.
- MBTI KB chưa được audit sâu như OCEAN.
- Linguistic correlates là weak evidence, cần caveat trong prompt/reranking.
- BFI-2 anchors là paraphrase project-authored, không phải raw manual text.
- Few-shot examples vẫn cùng collection, dù đã được category-filtered khỏi default retrieval path.

## 15. Khuyến Nghị Tiếp Theo

Các bước nên làm sau v3:

1. Tạo retrieval QA set khoảng `100` queries chia theo state, trait, marker, abstention, aggregation.
2. Chạy ablation `KB v2 vs KB v3` trên PersonalityEvd sample.
3. Đo riêng evidence F1, state consistency, grounding, classification macro F1.
4. Thêm category-aware reranking để `evidence_mapping_example` không lấn át trait definitions khi query là pure definition.
5. Audit lại `quality_tier=C` và thay dần bằng A/B sources.
6. Log `kb_version`, `kb_hash`, `collection_name` trong mọi experiment output.

## 16. Kết Luận

KB hiện tại được chọn và xây dựng để giải quyết đúng điểm yếu của bài toán: các dataset không có
knowledge base tâm lý học kèm theo, trong khi nghiên cứu cần explanation có thể kiểm chứng. Cấu trúc
v3 kết hợp tri thức học thuật, markers hành vi, linguistic correlates, PersonalityEvd train/valid
evidence mappings, abstention rules và state-trait aggregation rules.

Quyết định quan trọng nhất là không coi KB như một kho văn bản lớn, mà coi KB như một tập knowledge
atoms có provenance và metadata. Cách này làm retrieval sạch hơn, explanation traceable hơn, và
giúp RAG-XPR có cơ sở khoa học rõ hơn khi map evidence sang state và trait.
