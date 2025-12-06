# Pydantic AI + pydantic-graph 移行完了

## 変更概要

ai-improve-issue を **Pydantic AI** + **pydantic-graph** ベースの実装に全面的に書き換えました。

pydantic-graphの `BaseNode` を継承したノード定義と、`Graph` による実行フローで実装しています。

## 主な変更点

### 1. アーキテクチャ

- **Pydantic AI**: 型安全なLLM呼び出し、構造化出力対応
- **pydantic-graph**: ワークフローグラフによる柔軟なフロー制御
- **マルチプロバイダー対応**: Google/OpenAI/Anthropic を設定で切り替え可能

### 2. 新規ファイル

```
src/ai_improve_issue/
├── models.py          # Pydantic models定義（型安全性）
├── config.py          # 設定管理（YAML + 環境変数）
├── llm_factory.py     # LLM Agent ファクトリー（プロバイダー抽象化）
├── rag_client.py      # RAG機能（Embedding + Vector検索）
├── graph_nodes.py     # 処理ノード定義（各ステップの実装）
├── workflow_graph.py  # ワークフローグラフ（フロー制御）
└── main.py           # エントリポイント（CLI + 実行制御）
```

### 3. 設定ファイル拡張

`.ai_improve_issue.yml` に以下を追加：

```yaml
llm:
  provider: google  # google, openai, anthropic
  model: gemini-2.0-flash-exp
  temperature: 0.7
  max_tokens: 5000

providers:
  google:
    api_key_env: GEMINI_API_KEY
    models: [...]
  openai:
    api_key_env: OPENAI_API_KEY
    models: [...]
  anthropic:
    api_key_env: ANTHROPIC_API_KEY
    models: [...]

rag:
  enabled: true
  embedding_dimensions: 256
  similarity_threshold: 0.5
  top_k: 3
```

### 3. フロー実行（pydantic-graph）

**pydantic-graph による実装:**

各ノードは `BaseNode[GraphState]` を継承し、`run` メソッドで次のノードまたは `End` を返します。

```python
@dataclass
class InputValidationNode(BaseNode[GraphState]):
    async def run(self, ctx: GraphRunContext[GraphState]) -> RAGSearchNode | TemplateDetectionNode:
        # 入力検証処理
        if ctx.state.is_rag_enabled:
            return RAGSearchNode()
        else:
            return TemplateDetectionNode()
```

**通常モード実行フロー（Edgeラベル付き）:**

```
1. InputValidationNode（入力バリデーション）
   ├─ [RAG検索実行] → RAGSearchNode
   └─ [テンプレート判定へ] → TemplateDetectionNode
        ↓ [類似Issue検索完了]
2. RAGSearchNode（Voyage AI + Qdrant）※条件付き
        ↓
3. TemplateDetectionNode（LLMテンプレート判定）
        ↓ [テンプレート判定完了]
4. ContentGenerationNode（LLM文章生成）
        ↓ [LLM文章生成完了]
5. CommentFormattingNode（コメントフォーマット）
        ↓ [コメント作成完了]
        End(formatted_comment)
```

**Edgeラベル:**
- `RAG検索実行` - Voyage AI Embedding + Qdrant Vector検索
- `テンプレート判定へ` - RAGスキップ、直接テンプレート判定
- `類似Issue検索完了` - 類似Issue取得済み
- `テンプレート判定完了` - LLMテンプレート分類完了
- `LLM文章生成完了` - LLM本文生成完了
- `コメント作成完了` - Markdownコメント完成

**Graph実行:**

```python
graph = Graph(nodes=[...], state_type=GraphState)
result = await graph.run(InputValidationNode(), state=state)
formatted_comment = result.output
```

**Mermaid図の生成:**

```python
# pydantic-graphの機能で図を自動生成
mermaid_code = graph.mermaid_code(start_node=InputValidationNode)
print(mermaid_code)
```

### 5. プロバイダー切り替え

設定ファイルで `llm.provider` と `llm.model` を変更するだけで切り替え可能：

```yaml
# Google Gemini
llm:
  provider: google
  model: gemini-2.0-flash-exp

# OpenAI GPT
llm:
  provider: openai
  model: gpt-4o-mini

# Anthropic Claude
llm:
  provider: anthropic
  model: claude-3-5-sonnet-latest
```

## 互換性

- 既存の環境変数は全て互換
- CLI引数は完全互換（`--dry-run`, `--index-issues`, `--update-single-issue`）
- 実行モードは変更なし

## 依存ライブラリ

```toml
dependencies = [
    "pydantic>=2.10.0",
    "pydantic-ai>=0.0.14",
    "pydantic-graph>=0.4.1",
    "voyageai>=0.2.3",
    "qdrant-client==1.16.*",
    "langchain-text-splitters>=0.3.0",
    "pyyaml>=6.0",
]
```

## 実行方法

変更なし。既存のコマンドがそのまま動作します：

```bash
# 通常実行
ISSUE_NUMBER=1 ISSUE_TITLE="..." ISSUE_BODY="..." uv run ai-improve-issue

# ドライラン
uv run ai-improve-issue --dry-run

# RAGインデックス作成
uv run ai-improve-issue --index-issues

# 単一Issue更新
uv run ai-improve-issue --update-single-issue 123
```

## 今後の拡張

1. **カスタムフロー**: `workflow_graph.py` でノード追加・削除が容易
2. **プロバイダー追加**: `llm_factory.py` にプロバイダー設定を追加
3. **並列処理**: pydantic-graph の機能を活用して並列実行
4. **監視**: 各ノードの実行状況をログ・メトリクスとして収集

## バックアップ

旧実装は `main_old.py` として保存されています。
