"""RAG機能（Embedding + Vector検索）"""

import uuid

import voyageai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from .models import SimilarIssue


class VoyageEmbeddingClient:
    """Voyage AI Embeddingクライアント"""

    def __init__(self, api_key: str, model: str = "voyage-3.5-lite"):
        self.client = voyageai.Client(api_key=api_key)
        self.model = model

    def generate_embedding(self, text: str, dimensions: int = 256) -> list[float]:
        """テキストのEmbeddingを生成"""
        result = self.client.embed(
            texts=[text], model=self.model, output_dimension=dimensions
        )
        return result.embeddings[0]

    def generate_embeddings_batch(
        self, texts: list[str], dimensions: int = 256
    ) -> list[list[float]]:
        """複数テキストのEmbeddingを一括生成"""
        result = self.client.embed(
            texts=texts, model=self.model, output_dimension=dimensions
        )
        return result.embeddings


class QdrantSearchClient:
    """Qdrant検索クライアント"""

    def __init__(self, url: str, api_key: str, collection_name: str):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name

    def ensure_collection(self, vector_size: int = 256):
        """コレクションが存在することを確認、なければ作成"""
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"Collection '{self.collection_name}' created")
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="issue_number",
                field_schema=PayloadSchemaType.INTEGER,
            )

    def search_similar_issues(
        self,
        query_vector: list[float],
        limit: int = 3,
        exclude_issue_number: int | None = None,
    ) -> list[SimilarIssue]:
        """類似Issue検索"""
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit * 5,
        )

        points = getattr(response, "points", [])
        if not points:
            return []

        # Issueごとに最高スコアのチャンクを集約
        issue_map: dict[int, SimilarIssue] = {}
        for result in points:
            issue_num = result.payload.get("issue_number")
            if exclude_issue_number is not None and issue_num == exclude_issue_number:
                continue

            if (
                issue_num not in issue_map
                or result.score > issue_map[issue_num]["similarity"]
            ):
                issue_body = result.payload.get(
                    "issue_body_chunk"
                ) or result.payload.get("issue_body", "")

                issue_map[issue_num] = SimilarIssue(
                    issue_number=issue_num,
                    issue_title=result.payload.get("issue_title", ""),
                    issue_body=issue_body[:500],
                    state=result.payload.get("state", ""),
                    url=result.payload.get("url", ""),
                    similarity=result.score,
                )

        similar_issues = sorted(
            issue_map.values(), key=lambda x: x.similarity, reverse=True
        )[:limit]

        return similar_issues

    def upsert_issue_chunks(
        self,
        issue_number: int,
        chunks: list[str],
        vectors: list[list[float]],
        title: str,
        state: str,
        url: str,
        labels: list[str],
    ):
        """Issueをチャンク分割してインデックスに登録または更新"""
        # 既存のチャンクを削除
        ids_to_delete: list[str] = []
        offset: dict | None = None

        while True:
            existing_points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="issue_number",
                            match=MatchValue(value=issue_number),
                        )
                    ]
                ),
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )

            if not existing_points:
                break

            ids_to_delete.extend(str(point.id) for point in existing_points)

            if next_offset is None:
                break

            offset = next_offset

        if ids_to_delete:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=ids_to_delete),
            )

        # 新しいチャンクを登録
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "issue_number": issue_number,
                    "chunk_index": i,
                    "issue_title": title,
                    "issue_body_chunk": chunk,
                    "state": state,
                    "url": url,
                    "labels": labels,
                },
            )
            points.append(point)

        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"Issue #{issue_number} indexed with {len(chunks)} chunks")


def create_issue_chunks(issue_title: str, issue_body: str) -> list[str]:
    """Issue本文をチャンク分割"""
    full_text = f"{issue_title}\n\n{issue_body}"

    if len(full_text) <= 400:
        return [full_text]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["。", "\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_text(full_text)
    return chunks
