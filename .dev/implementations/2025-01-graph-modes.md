# issue-2025: 全モードをGraphベースに統一

## 対応内容

すべての実行モード（通常、RAGインデックス、単一Issue更新）をpydantic-graphベースで統一実装しました。

### 対応範囲

#### 1. graph_nodes.py - 新規ノード追加

**RAGインデックスモード用:**
- `RAGIndexState` - インデックス処理の状態管理（Issue一覧、カウンタ）
- `RAGIndexInitNode` - Qdrant接続確認、コレクション初期化
- `RAGIndexProcessNode` - Issue単位のループ処理（チャンク分割→Embedding生成→Upsert）
- `RAGIndexCompleteNode` - 完了ログ出力、結果返却（終了ノード）

**単一Issue更新モード用:**
- `SingleIssueUpdateState` - 単一Issue更新の状態管理
- `SingleIssueUpdateNode` - 1つのIssueをインデックス更新（終了ノード）

**フロー図:**

```
RAGインデックスモード:
RAGIndexInitNode → [インデックス処理開始] → RAGIndexProcessNode
    ↓ [次のIssue処理]
RAGIndexProcessNode ← ← ← ← ← ← ← (ループ)
    ↓ [インデックス完了]
RAGIndexCompleteNode → [インデックス完了] → End(dict)

単一Issue更新モード:
SingleIssueUpdateNode → [Issue更新完了] → End(dict)
```

#### 2. workflow_graph.py - グラフ生成関数追加

```python
# RAGインデックス用グラフ生成
create_rag_index_graph(settings, env_config) -> Graph[RAGIndexState, None, dict]
create_rag_index_state(issues, settings, env_config) -> RAGIndexState

# 単一Issue更新用グラフ生成
create_single_issue_update_graph() -> Graph[SingleIssueUpdateState, None, dict]
create_single_issue_update_state(issue, settings, env_config) -> SingleIssueUpdateState
```

#### 3. main.py - モード処理の統一

**--index-issues モード:**
```python
# 古い: for ループで直接 Voyage AI/Qdrant 呼び出し
# 新しい: RAGIndexInitNode() から Graph.run() で実行
graph = create_rag_index_graph(settings, env_config)
state = create_rag_index_state(issues, settings, env_config)
result = await graph.run(RAGIndexInitNode(), state=state)
```

**--update-single-issue モード:**
```python
# 古い: 直接 Voyage AI/Qdrant 呼び出し
# 新しい: SingleIssueUpdateNode() から Graph.run() で実行
graph = create_single_issue_update_graph()
state = create_single_issue_update_state(issue, settings, env_config)
result = await graph.run(SingleIssueUpdateNode(), state=state)
```

**通常モード（コメント投稿後のRAG更新）:**
```python
# 古い: 直接 Voyage AI/Qdrant 呼び出し
# 新しい: SingleIssueUpdateNode() で更新
current_issue = IssueData(issue_number, title, body, ...)
graph = create_single_issue_update_graph()
state = create_single_issue_update_state(current_issue, settings, env_config)
result = await graph.run(SingleIssueUpdateNode(), state=state)
```

### 対応範囲外

- 既存API互換性の維持（不要）
- 旧実装（main_old.py）の更新

## 技術的な注意点

1. **ノードのループ処理**
   - RAGIndexProcessNode は条件分岐で自分自身を返すことでループを実現
   - 各イテレーション後に `current_index` インクリメント

2. **Edge ラベル**
   - RAGインデックス: `インデックス処理開始` → `次のIssue処理` (ループ) → `インデックス完了`
   - 単一Issue更新: `Issue更新完了`

3. **非同期処理**
   - main.py の `async_main()` 内で個別に `asyncio.run()` を呼ばずに、すべてのグラフ実行を await で待機
   - RAGインデックス/単一更新モードでは別途 `asyncio.run()` が必要（sys.exit()のため）

4. **戻り値**
   - RAGインデックス完了: `{"success": N, "total": M}`
   - 単一Issue更新: `{"issue_number": N, "chunks_indexed": M}`

## 確認してほしいこと

- [ ] RAGインデックスモード（--index-issues）の動作確認
- [ ] 単一Issue更新モード（--update-single-issue N）の動作確認
- [ ] 通常モード（コメント投稿 + RAG更新）の動作確認
- [ ] Voyage AI、Qdrant ラベル表示の確認

## 変更した関数/クラス

### graph_nodes.py
- 新規: RAGIndexState, RAGIndexInitNode, RAGIndexProcessNode, RAGIndexCompleteNode
- 新規: SingleIssueUpdateState, SingleIssueUpdateNode

### workflow_graph.py
- 新規: create_rag_index_graph(), create_rag_index_state()
- 新規: create_single_issue_update_graph(), create_single_issue_update_state()

### main.py
- 修正: index_all_issues() - グラフベースに統一
- 修正: update_single_issue() - グラフベースに統一
- 修正: async_main() 内 RAG更新 - グラフベースに統一
- 削除: 直接的な Voyage AI/Qdrant クライアント使用（→ノードに移動）

### workflow_graph.py インポート
- 追加: RAGIndexState, RAGIndexInitNode, RAGIndexProcessNode, RAGIndexCompleteNode
- 追加: SingleIssueUpdateState, SingleIssueUpdateNode
- 追加: IssueData型
