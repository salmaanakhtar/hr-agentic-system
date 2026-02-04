from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import logging
import os
import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from . import VectorTool, ToolResult


class VectorSearchInput(BaseModel):
    query_vector: List[float] = Field(..., description="Query vector for similarity search")
    table_name: str = Field(..., description="Table containing vector data")
    vector_column: str = Field(default="embedding", description="Column name for vectors")
    limit: int = Field(default=10, description="Maximum number of results to return")
    threshold: Optional[float] = Field(default=None, description="Similarity threshold (0-1)")


class VectorSearchOutput(BaseModel):
    results: List[Dict[str, Any]] = Field(..., description="Search results with similarity scores")
    total_results: int = Field(..., description="Total number of results found")


class VectorStoreInput(BaseModel):
    vectors: List[List[float]] = Field(..., description="Vectors to store")
    metadata: List[Dict[str, Any]] = Field(..., description="Metadata for each vector")
    table_name: str = Field(..., description="Table to store vectors in")
    vector_column: str = Field(default="embedding", description="Column name for vectors")


class VectorStoreOutput(BaseModel):
    stored_count: int = Field(..., description="Number of vectors stored")


class PgVectorTool(VectorTool[BaseModel, BaseModel]):

    def __init__(self, name: str, description: str, database_url: str):
        super().__init__(name, description, database_url)
        self.engine = None
        self.session_factory = None

    async def connect(self) -> None:
        try:
            from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
            self.engine = create_async_engine(
                self.connection_string,
                echo=False,
                pool_pre_ping=True
            )
            self.session_factory = async_sessionmaker(
                self.engine, class_=AsyncSession, expire_on_commit=False
            )
            self.logger.info("Vector database connection established")
        except Exception as e:
            self.logger.error(f"Failed to connect to vector database: {e}")
            raise

    async def disconnect(self) -> None:
        if self.engine:
            await self.engine.dispose()
            self.logger.info("Vector database connection closed")

    async def search_similar(self, query_embedding: List[float],
                           limit: int = 10, threshold: float = 0.7) -> List[Dict[str, Any]]:
        return await self.similarity_search(
            query_vector=query_embedding,
            table_name="policy_documents",
            vector_column="embedding",
            limit=limit,
            threshold=threshold
        )

    async def store_embedding(self, document_id: str, embedding: List[float],
                            metadata: Dict[str, Any]) -> str:
        metadata_with_id = {**metadata, "document_id": document_id}

        stored_count = await self.store_vectors(
            vectors=[embedding],
            metadata=[metadata_with_id],
            table_name="policy_documents"
        )

        return document_id if stored_count > 0 else ""

    async def delete_embedding(self, document_id: str) -> bool:
 
        async with self.session_factory() as session:
            try:
                query = """
                DELETE FROM policy_documents
                WHERE metadata->>'document_id' = :document_id
                """

                result = await session.execute(text(query), {"document_id": document_id})
                await session.commit()

                return result.rowcount > 0

            except Exception as e:
                await session.rollback()
                self.logger.error(f"Embedding deletion failed: {e}")
                return False

    async def similarity_search(self, query_vector: List[float], table_name: str,
                              vector_column: str = "embedding", limit: int = 10,
                              threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        async with self.session_factory() as session:
            try:

                vector_str = f"[{','.join(map(str, query_vector))}]"

                query = f"""
                SELECT *, ({vector_column} <=> :query_vector) as similarity
                FROM {table_name}
                WHERE {vector_column} IS NOT NULL
                """

                if threshold is not None:
                    query += f" AND ({vector_column} <=> :query_vector) >= :threshold"

                query += f" ORDER BY {vector_column} <=> :query_vector LIMIT :limit"

                result = await session.execute(
                    text(query),
                    {
                        "query_vector": vector_str,
                        "threshold": threshold,
                        "limit": limit
                    }
                )

                rows = result.fetchall()
                return [dict(row._mapping) for row in rows]

            except Exception as e:
                self.logger.error(f"Vector search failed: {e}")
                raise

    async def store_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]],
                          table_name: str, vector_column: str = "embedding") -> int:
        async with self.session_factory() as session:
            try:

                data = []
                for vector, meta in zip(vectors, metadata):
                    vector_str = f"[{','.join(map(str, vector))}]"
                    row = {vector_column: vector_str, **meta}
                    data.append(row)

                if not data:
                    return 0


                columns = list(data[0].keys())
                placeholders = [f":{col}" for col in columns]
                query = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
                """


                await session.execute(text(query), data)
                await session.commit()

                return len(data)

            except Exception as e:
                await session.rollback()
                self.logger.error(f"Vector storage failed: {e}")
                raise

    async def execute(self, input_data: BaseModel) -> ToolResult:
        try:
            if isinstance(input_data, VectorSearchInput):
                results = await self.similarity_search(
                    input_data.query_vector,
                    input_data.table_name,
                    input_data.vector_column,
                    input_data.limit,
                    input_data.threshold
                )
                output = VectorSearchOutput(results=results, total_results=len(results))
                return self.create_result(True, output.dict())
            elif isinstance(input_data, VectorStoreInput):
                stored_count = await self.store_vectors(
                    input_data.vectors,
                    input_data.metadata,
                    input_data.table_name,
                    input_data.vector_column
                )
                output = VectorStoreOutput(stored_count=stored_count)
                return self.create_result(True, output.dict())
            else:
                return self.create_result(False, error="Unsupported input type")
        except Exception as e:
            return self.create_result(False, error=str(e))

    def get_input_schema(self):
        return BaseModel 

    def get_output_schema(self):
        return BaseModel  


class HRVectorTool(PgVectorTool):

    def __init__(self, database_url: Optional[str] = None):
        db_url = database_url or os.getenv('VECTOR_DB_URL') or os.getenv('DATABASE_URL')
        if not db_url:
            raise ValueError("Vector database URL must be provided or set in VECTOR_DB_URL/DATABASE_URL environment variable")

        super().__init__(
            name="hr_vector_search",
            description="Vector search tool for HR documents and policies",
            database_url=db_url
        )

    async def search_policies(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        return await self.similarity_search(
            query_vector=query_vector,
            table_name="policy_documents",
            vector_column="embedding",
            limit=limit
        )

    async def search_employee_documents(self, query_vector: List[float],
                                      employee_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        async with self.session_factory() as session:
            try:
                vector_str = f"[{','.join(map(str, query_vector))}]"

                query = """
                SELECT *, (embedding <=> :query_vector) as similarity
                FROM employee_documents
                WHERE employee_id = :employee_id
                AND embedding IS NOT NULL
                ORDER BY embedding <=> :query_vector
                LIMIT :limit
                """

                result = await session.execute(
                    text(query),
                    {
                        "query_vector": vector_str,
                        "employee_id": employee_id,
                        "limit": limit
                    }
                )

                rows = result.fetchall()
                return [dict(row._mapping) for row in rows]

            except Exception as e:
                self.logger.error(f"Employee document search failed: {e}")
                raise

    async def store_policy_embedding(self, policy_id: int, content: str,
                                   embedding: List[float]) -> bool:
        try:
            await self.store_vectors(
                vectors=[embedding],
                metadata=[{
                    "policy_id": policy_id,
                    "content": content,
                    "created_at": "NOW()"
                }],
                table_name="policy_documents"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to store policy embedding: {e}")
            return False